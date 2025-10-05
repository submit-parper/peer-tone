import os
import json
import queue
import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import tiktoken
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# ========== 配置类 ==========
@dataclass
class Config:
    """配置类，集中管理所有配置项"""
    # API配置
    api_keys: List[str]
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "Qwen/QwQ-32B"
    
    # 文件路径配置
    input_dir: str = "/merged_prompts"
    output_file: str = "/tone_output.json"
    state_file: str = "/.state_tone_2009-2024"
    
    # 处理配置
    max_workers: int = 10
    max_retries: int = 3
    request_interval: float = 2.0
    
    # Token配置
    max_token_length: int = 30000
    chunk_size: int = 15000
    
    # 日志配置
    log_level: str = "INFO"
    log_file: str = "/processing.log"
    
    # 动态max_tokens配置
    dynamic_max_tokens: bool = True
    max_tokens_options: Dict[str, int] = None
    base_generation_params: Dict[str, Any] = None

def load_config(config_file: str = "/config.json") -> Config:
    """从配置文件加载配置"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 验证必需的配置项
        if 'api_keys' not in config_data or not config_data['api_keys']:
            raise ValueError("配置文件中缺少API密钥")
        
        return Config(**config_data)
    except FileNotFoundError:
        print(f"配置文件 {config_file} 不存在，使用默认配置")
        return Config(api_keys=[])
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return Config(api_keys=[])

# ========== 日志配置 ==========
def setup_logging(config: Config):
    """设置日志配置"""
    # 确保日志目录存在
    log_dir = os.path.dirname(config.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ========== Token编码器 ==========
try:
    TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    TOKEN_ENCODER = tiktoken.get_encoding("gpt2")

def count_tokens(text: str) -> int:
    """计算文本的token数量"""
    return len(TOKEN_ENCODER.encode(text))

# ========== 动态max_tokens计算 ==========
def calculate_dynamic_max_tokens(content: str, config: Config) -> int:
    """根据文件内容动态计算max_tokens"""
    if not config.dynamic_max_tokens:
        # 如果不启用动态处理，使用默认值
        return config.base_generation_params.get('max_tokens', 1500) if config.base_generation_params else 1500
    
    # 计算文件token数
    total_tokens = count_tokens(content)
    
    # 估算MD&A部分（假设占70%）
    mda_tokens = int(total_tokens * 0.7)
    
    # 根据文件大小选择max_tokens策略
    if mda_tokens <= 3000:
        # 小文件：使用保守设置
        max_tokens = config.max_tokens_options.get('conservative', 6855)
    elif mda_tokens <= 8000:
        # 中等文件：使用平衡设置
        max_tokens = config.max_tokens_options.get('balanced', 8569)
    elif mda_tokens <= 15000:
        # 大文件：使用全面设置
        max_tokens = config.max_tokens_options.get('comprehensive', 11426)
    else:
        # 超大文件：使用最大设置
        max_tokens = config.max_tokens_options.get('maximum', 21610)
    
    return max_tokens

# ========== Prompt 模板 ==========
PROMPT_TEMPLATE = (
    "你是一位专业的财务语调分析专家。以下是某公司年报信息和MD&A文本，请从中识别所有表达经营成果或业绩表现的句子，"
    "并根据公司财务指标判断其语调类型。\n\n"
    "【公司信息】和【MD&A文本】如下：\n"
)

TONE_EXPLANATION = (
    "语调分类说明：\n"
    "1（异常积极）：业绩差但语调非常乐观；\n"
    "0（正常积极）：语调积极，与业绩相符；\n"
    "-1（消极）：语调消极，或业绩差描述合理。\n\n"
    "请只输出一个 JSON 数组，每个元素为一个判断结果，对象字段包括：\n"
    "  \"sentence\": 原句内容；\n"
    "  \"tone_class\": 语调分类（1/0/-1）；\n"
    "  \"reason\": 判断理由，简洁说明语调与业绩是否匹配。\n"
)

# ========== 分块处理 ==========
def split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    """将文本按token数量分块"""
    tokens = TOKEN_ENCODER.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(TOKEN_ENCODER.decode(chunk_tokens))
    return chunks

def smart_split_text(text: str, max_tokens: int, chunk_size: int) -> str:
    """智能分块处理，优先保留重要内容"""
    if count_tokens(text) <= max_tokens:
        return text
    
    # 如果超过token限制，进行分块处理
    chunks = split_text_into_chunks(text, chunk_size)
    
    # 优先保留前两个块（通常包含最重要的信息）
    if len(chunks) >= 2:
        return ''.join(chunks[:2])
    else:
        return chunks[0] if chunks else text

# ========== 状态管理 ==========
class ProcessingState:
    """处理状态管理类"""
    
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.lock = threading.Lock()
        self.processed = set()
        self.failed = set()
        self._load_state()
    
    def _load_state(self):
        """加载处理状态"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed = set(data.get('processed', []))
                    self.failed = set(data.get('failed', []))
                logger.info(f"加载状态：已处理 {len(self.processed)} 个文件，失败 {len(self.failed)} 个文件")
        except Exception as e:
            logger.warning(f"加载状态文件失败: {e}")
    
    def _save_state(self):
        """保存处理状态"""
        try:
            # 确保目录存在
            state_dir = os.path.dirname(self.state_file)
            if state_dir and not os.path.exists(state_dir):
                os.makedirs(state_dir)
            
            temp = self.state_file + '.tmp'
            with open(temp, 'w', encoding='utf-8') as f:
                json.dump({
                    'processed': list(self.processed), 
                    'failed': list(self.failed)
                }, f, ensure_ascii=False, indent=2)
            os.replace(temp, self.state_file)
        except Exception as e:
            logger.error(f"保存状态文件失败: {e}")
    
    def mark_processed(self, fn: str):
        """标记文件为已处理"""
        with self.lock:
            self.processed.add(fn)
            self.failed.discard(fn)
            self._save_state()
    
    def mark_failed(self, fn: str):
        """标记文件为处理失败"""
        with self.lock:
            self.failed.add(fn)
            self._save_state()
    
    def is_processed(self, fn: str) -> bool:
        """检查文件是否已处理"""
        return fn in self.processed
    
    def get_progress(self) -> Dict[str, int]:
        """获取处理进度"""
        return {
            'processed': len(self.processed),
            'failed': len(self.failed)
        }

# ========== 核心处理类 ==========
class ToneClassifier:
    """语调分类器主类"""
    
    def __init__(self, config: Config):
        self.config = config
        self.state = ProcessingState(config.state_file)
        self.results = self._load_results()
        self.api_queue = queue.Queue()
        
        # 初始化API密钥队列
        for key in config.api_keys:
            self.api_queue.put(key)
        
        self.lock = threading.Lock()
        self.counter = 0
        self.start_time = time.time()
        
        logger.info(f"初始化完成，API密钥数量: {len(config.api_keys)}")
        logger.info(f"动态max_tokens: {'启用' if config.dynamic_max_tokens else '禁用'}")
    
    def _load_results(self) -> Dict[str, Any]:
        """加载已有结果"""
        if os.path.exists(self.config.output_file):
            try:
                with open(self.config.output_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.info(f"加载已有结果: {len(results)} 个文件")
                return results
            except Exception as e:
                logger.error(f"加载结果文件失败: {e}")
        return {}
    
    def _save_results(self):
        """保存结果到文件"""
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(self.config.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 合并已有结果
            merged = {}
            if os.path.exists(self.config.output_file):
                try:
                    with open(self.config.output_file, 'r', encoding='utf-8') as f:
                        merged = json.load(f)
                except:
                    merged = {}
            
            merged.update(self.results)
            
            # 使用临时文件保存
            tmp = self.config.output_file + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.config.output_file)
            
            self.results = merged
            logger.info(f"保存结果成功，总计 {len(merged)} 个文件")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def _get_client(self) -> OpenAI:
        """获取API客户端"""
        key = self.api_queue.get()
        client = OpenAI(api_key=key, base_url=self.config.base_url)
        self.api_queue.put(key)
        return client
    
    def parse_response(self, text: str) -> Optional[List[Dict]]:
        """解析API响应"""
        try:
            # 清理响应文本
            text = text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            
            data = json.loads(text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'results' in data:
                return data['results']
        except Exception as e:
            logger.debug(f"解析响应失败: {e}, 响应内容: {text[:200]}...")
        return None
    
    def process_file(self, path: str, retry: int = 0) -> bool:
        """处理单个文件"""
        fn = os.path.basename(path)
        
        # 检查是否已处理
        if self.state.is_processed(fn):
            logger.debug(f"文件已处理，跳过: {fn}")
            return True
        
        try:
            logger.info(f"开始处理文件: {fn} (重试次数: {retry})")
            
            # 读取文件内容
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 计算文件token数
            total_tokens = count_tokens(content)
            logger.info(f"文件 {fn} 总token数: {total_tokens:,}")
            
            # 智能分块处理
            processed_content = smart_split_text(
                content, 
                self.config.max_token_length, 
                self.config.chunk_size
            )
            
            # 动态计算max_tokens
            dynamic_max_tokens = calculate_dynamic_max_tokens(content, self.config)
            logger.info(f"文件 {fn} 动态max_tokens: {dynamic_max_tokens:,}")
            
            # 构建prompt
            prompt = PROMPT_TEMPLATE + processed_content + "\n\n" + TONE_EXPLANATION
            
            # 调用API
            client = self._get_client()
            
            # 准备生成参数
            generation_kwargs = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": "你是一个严谨的财务语调分析专家，仅根据提供信息判断语调。"},
                    {"role": "user", "content": prompt}
                ]
            }
            
            # 使用动态max_tokens
            if self.config.base_generation_params:
                # 复制基础参数
                generation_kwargs.update(self.config.base_generation_params)
                # 覆盖max_tokens为动态计算的值
                generation_kwargs['max_tokens'] = dynamic_max_tokens
            else:
                # 默认参数
                generation_kwargs.update({
                    "temperature": 0,
                    "top_p": 0.9,
                    "presence_penalty": 0.1,
                    "frequency_penalty": 0.1,
                    "max_tokens": dynamic_max_tokens
                })
            
            response = client.chat.completions.create(**generation_kwargs)
            
            raw_response = response.choices[0].message.content.strip()
            parsed_result = self.parse_response(raw_response)
            
            # 处理解析结果
            if not parsed_result:
                if retry < self.config.max_retries:
                    wait_time = 2 ** retry
                    logger.warning(f"解析失败，{wait_time}秒后重试: {fn}")
                    time.sleep(wait_time)
                    return self.process_file(path, retry + 1)
                else:
                    logger.error(f"解析失败，达到最大重试次数: {fn}")
                    self.state.mark_failed(fn)
                    return False
            
            # 保存结果
            with self.lock:
                self.results[fn] = parsed_result
                self.state.mark_processed(fn)
                self.counter += 1
                
                # 定期保存
                if self.counter % 10 == 0:
                    logger.info(f"已处理 {self.counter} 个文件，保存结果...")
                    self._save_results()
            
            logger.info(f"成功处理文件: {fn}")
            return True
            
        except Exception as e:
            logger.error(f"处理文件异常 {fn}: {e}")
            if retry < self.config.max_retries:
                wait_time = 2 ** retry
                logger.info(f"{wait_time}秒后重试: {fn}")
                time.sleep(wait_time)
                return self.process_file(path, retry + 1)
            else:
                self.state.mark_failed(fn)
                return False
    
    def run(self):
        """运行主处理流程"""
        # 获取所有文件
        input_path = Path(self.config.input_dir)
        if not input_path.exists():
            logger.error(f"输入目录不存在: {self.config.input_dir}")
            return
        
        files = list(input_path.rglob("*.txt"))
        logger.info(f"找到 {len(files)} 个文件在 {self.config.input_dir}")
        
        if not files:
            logger.warning("未找到.txt文件，请检查input_dir路径")
            return
        
        # 过滤已处理的文件
        files_to_process = [f for f in files if not self.state.is_processed(f.name)]
        total_files = len(files_to_process)
        logger.info(f"需要处理 {total_files} 个文件")
        logger.info(f"进度显示频率：每处理10个文件显示一次进度")
        
        # 并发处理
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 提交所有任务
            future_to_file = {executor.submit(self.process_file, str(f)): f for f in files_to_process}
            
            # 处理完成的任务
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        successful += 1
                    else:
                        failed += 1
                    
                    # 显示进度
                    total_processed = successful + failed
                    if total_processed % 10 == 0:
                        elapsed = time.time() - self.start_time
                        rate = total_processed / elapsed if elapsed > 0 else 0
                        remaining = total_files - total_processed
                        logger.info(f"进度: {total_processed}/{total_files} "
                                  f"(剩余: {remaining}, 成功: {successful}, 失败: {failed}, 速率: {rate:.2f} 文件/秒)")
                
                except Exception as e:
                    logger.error(f"处理文件异常 {file_path}: {e}")
                    failed += 1
        
        # 最终保存
        logger.info("处理完成，保存最终结果...")
        self._save_results()
        
        # 输出统计信息
        progress = self.state.get_progress()
        elapsed = time.time() - self.start_time
        logger.info(f"处理完成！总计: {progress['processed']} 成功, {progress['failed']} 失败, "
                   f"耗时: {elapsed:.2f} 秒")
        logger.info(f"最终进度: {total_processed}/{total_files} 文件已处理完成")

def main():
    """主函数"""
    print("开始语调分析处理（动态max_tokens版本）...")
    
    # 加载配置
    config = load_config()
    
    # 检查配置
    if not config.api_keys:
        print("错误：未配置API密钥，请检查config.json文件")
        return
    
    # 设置日志
    global logger
    logger = setup_logging(config)
    
    # 运行处理
    classifier = ToneClassifier(config)
    classifier.run()
    
    logger.info("处理完成！")

if __name__ == '__main__':
    main() 
