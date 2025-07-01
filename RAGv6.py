import hashlib
import torch
import os
import json
import re
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document  # 添加到文件顶部的导入部分
from langchain.llms.base import LLM
from threading import Thread
import queue
import subprocess
#import warnings
#import markdown
#from tkhtmlview import HTMLLabel
from datetime import datetime
import webbrowser
import requests
import json
from typing import Any, Dict, List, Optional, Union, Sequence, Literal
from dataclasses import dataclass
from langchain.callbacks.manager import CallbackManagerForLLMRun
#from langchain.schema import LLMResult
from langchain_community.llms import Ollama
from pydantic import Field
from langchain.prompts import PromptTemplate


class WorkingOpenAILLM(LLM):
    """完全兼容的OpenAI LLM客户端"""
    
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.7):
        # 使用字典存储配置，避免Pydantic限制
        config = {
            'base_url': base_url.rstrip('/'),
            'api_key': api_key,
            'model': model,
            'temperature': temperature
        }
        super().__init__(**config)
    
    @property
    def _llm_type(self) -> str:
        return "working_openai"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # 从实例属性获取配置
        base_url = getattr(self, 'base_url', 'https://api.openai.com/v1')
        api_key = getattr(self, 'api_key', '')
        model = getattr(self, 'model', 'gpt-3.5-turbo')
        temperature = getattr(self, 'temperature', 0.7)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        max_tokens = kwargs.get('max_tokens', 256)
        temp = kwargs.get('temperature', temperature)
        
        # 判断API格式
        if 'gpt-3.5' in model or 'gpt-4' in model:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temp,
                "stop": stop or []
            }
            endpoint = "/chat/completions"
        else:
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temp,
                "stop": stop or []
            }
            endpoint = "/completions"
        
        try:
            response = requests.post(
                f"{base_url}{endpoint}",
                headers=headers,
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                if 'message' in result['choices'][0]:
                    return result['choices'][0]['message']['content']
                else:
                    return result['choices'][0]['text']
            return ""
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return ""

# 如果上面还不行，使用这个完全不继承LLM的版本
class StandaloneLLM:
    """独立的LLM实现，不继承LangChain基类"""
    
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.7):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """直接调用接口"""
        return self.generate(prompt, **kwargs)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        max_tokens = kwargs.get('max_tokens', 256)
        temp = kwargs.get('temperature', self.temperature)
        stop = kwargs.get('stop', [])
        
        # 判断API格式
        if 'gpt-3.5' in self.model or 'gpt-4' in self.model:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temp,
                "stop": stop
            }
            endpoint = "/chat/completions"
        else:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temp,
                "stop": stop
            }
            endpoint = "/completions"
        
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                headers=headers,
                json=payload,
                timeout=3000
            )
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                if 'message' in result['choices'][0]:
                    return result['choices'][0]['message']['content']
                else:
                    return result['choices'][0]['text']
            return ""
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return ""

# LangChain兼容的包装器
class LangChainWrapper(LLM):
    
    standalone_llm: StandaloneLLM = Field(...)  # Declare the field with proper typing
    
    def __init__(self, standalone_llm: StandaloneLLM):
        super().__init__()  # Pass to parent constructor
        self.standalone_llm = standalone_llm
    
    @property
    def _llm_type(self) -> str:
        return "wrapped_llm"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self.standalone_llm.generate(prompt, stop=stop or [], **kwargs)

class WorkingLLMClient:
    """完全可用的LLM客户端"""
    
    def __init__(self):
        self.supported_backends = ['ollama', 'lmstudio', 'openai']
    
    def get_available_models(self, backend: Literal['ollama', 'lmstudio', 'openai'], 
                           base_url: Optional[str] = None, api_key: Optional[str] = None) -> List[str]:
        """获取指定后端的可用模型列表"""
        try:
            if backend == 'ollama':
                return self._get_ollama_models(base_url or "http://localhost:11434")
            elif backend == 'lmstudio':
                return self._get_lmstudio_models(base_url or "http://localhost:1234")
            elif backend == 'openai':
                return self._get_openai_models(api_key, base_url or "https://api.openai.com/v1")
            else:
                raise ValueError(f"不支持的后端: {backend}")
        except Exception as e:
            print(f"获取{backend}模型列表失败: {str(e)}")
            return []
    
    def _get_ollama_models(self, base_url: str) -> List[str]:
        """获取Ollama模型列表"""
        response = requests.get(f"{base_url}/api/tags", timeout=100)
        response.raise_for_status()
        models = response.json().get('models', [])
        return [model['name'] for model in models]
    
    def _get_lmstudio_models(self, base_url: str) -> List[str]:
        """获取LM Studio模型列表"""
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=100)
            response.raise_for_status()
            models = response.json().get('data', [])
            return [model['id'] for model in models]
        except:
            return ["local-model"]
    
    def _get_openai_models(self, api_key: Optional[str], base_url: str) -> List[str]:
        """获取OpenAI模型列表"""
        if not api_key:
            return ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "text-davinci-003"]
        
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(f"{base_url}/models", headers=headers, timeout=100)
            response.raise_for_status()
            models = response.json().get('data', [])
            return [model['id'] for model in models if any(keyword in model['id'] 
                    for keyword in ['gpt', 'text', 'davinci', 'curie', 'babbage', 'ada'])]
        except:
            return ["gpt-3.5-turbo", "gpt-4", "text-davinci-003"]
    def ensure_http_prefix(self, url: str) -> str:
        """确保URL包含http/https前缀"""
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = f"http://{url}"
        return url.rstrip('/')
    def llmClient(self, backend: Literal['ollama', 'lmstudio', 'openai'],
                  base_url: str, model: str, temperature: float = 0.7,
                  api_key: Optional[str] = None, **kwargs):
        """创建LLM客户端实例"""
        try:
            if backend == 'ollama':
                return Ollama(
                    base_url=self.ensure_http_prefix(base_url),
                    model=model,
                    temperature=temperature,
                    **kwargs
                )
            elif backend == 'lmstudio':
                # 使用独立实现 + 包装器
                standalone = StandaloneLLM(
                    base_url=self.ensure_http_prefix(f"{base_url}/v1"),
                    api_key="lm-studio",
                    model=model,
                    temperature=temperature
                )
                return LangChainWrapper(standalone)
                
            elif backend == 'openai':
                if not base_url:
                    raise ValueError("OpenAI需要API密钥")
                
                # 使用独立实现 + 包装器
                standalone = StandaloneLLM(
                    base_url= "https://api.openai.com/v1",
                    api_key=base_url,
                    model=model,
                    temperature=temperature
                )
                return LangChainWrapper(standalone)
            else:
                raise ValueError(f"不支持的后端: {backend}")
        except Exception as e:
            print(f"创建{backend}客户端失败: {str(e)}")
            raise


class PDFRAGApp:
    def __init__(self, root):
        self.torch = torch
        self.root = root
        self.root.title("高级PDF RAG系统 v6 (Ollama)")
        self.root.geometry("1300x850")
        self.conversation_html = ""
        # 初始化配置
        self._init_config()
        self._init_variables()
        self._init_device()
        self.client = WorkingLLMClient()
        # 创建GUI
        self.create_widgets()
        
        # 加载已有数据
        self.load_existing_data()
        
        # 启动消息处理
        self.root.after(100, self.process_messages)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    def on_close(self):
        if hasattr(self, self.vectorstore_path):
            self.save_vectorstore(self.vectorstore_path)
        self.save_hashes()
        self.root.destroy()

    def _init_config(self):
        """初始化配置路径"""
        print("初始化配置路径...")
        self.config_dir = "rag_config"
        os.makedirs(self.config_dir, exist_ok=True)
        self.vectorstore_path = os.path.join(self.config_dir, "vectorstore")
        self.hash_db_path = os.path.join(self.config_dir, "pdf_hashes.json")
        self.settings_path = os.path.join(self.config_dir, "app_settings.json")
        
    def _init_variables(self):
        """初始化变量"""
        print("初始化变量...")
        self.pdf_paths = []
        self.success_files = []
        self.failed_files = {}
        self.file_hashes = self.load_hashes()
        self.vectorstore = None
        self.qa_chain = None
        self.individual_vectorstores = {}
        self.conversation_history = []
        
        # 模型相关
        self.ollama_models = self.get_local_ollama_models()
        self.model_var = tk.StringVar(value=self.ollama_models[0] if self.ollama_models else "llama2")
        self.temp_var = tk.DoubleVar(value=0.7)
        self.max_tokens_var = tk.IntVar(value=2000)
        self.default_prompt = """You are an expert AI assistant specialized in providing accurate, context-based responses. Your role is to analyze the provided context and answer questions with precision and clarity.
        1. Start with a direct answer if possible
        2. Support with evidence from the context
        3. Note any limitations or assumptions
        4. Suggest follow-up questions if relevant
        5. When there are Priority Documents selected,  Do not mix the content from other documents """   
        
        # 消息系统
        self.message_queue = queue.Queue()
        self.llm_cache = {}

    def get_device(self):
        return self.device

    def _init_device(self):
        """初始化设备"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"运行设备: {self.device.upper()}")
        self.message_queue.put(("message", "系统", f"检测到运行设备: {self.device.upper()}"))

    def add_pdfs(self):
        """添加PDF文件到系统并处理
        
        1. 通过文件对话框选择PDF文件
        2. 检查是否已存在
        3. 为每个文件创建单独向量库
        4. 合并向量库
        5. 更新界面状态
        """
        # 通过文件对话框选择PDF文件
        files = filedialog.askopenfilenames(
            title="选择PDF文件",
            filetypes=[("PDF文件", "*.pdf"), ("所有文件", "*.*")]
        )
        
        if not files:
            return
        
        # 禁用按钮防止重复操作
        self.add_btn.config(state=tk.DISABLED)
        self.update_status("正在处理新增PDF文件...")
        
        # 在新线程中处理文件
        Thread(target=self._process_added_pdfs, args=(files,), daemon=True).start()

    def _process_added_pdfs(self, files):
        """实际处理新增PDF文件的逻辑"""
        try:
            added_count = 0
            skipped_count = 0
            failed_count = 0
            
            for file_path in files:
                # 检查是否已存在
                if file_path in self.pdf_paths:
                    print(f"文件已存在，跳过: {file_path}")
                    skipped_count += 1
                    continue
                
                # 处理PDF文件
                if self.process_single_pdf(file_path):
                    # 添加到列表
                    self.pdf_paths.append(file_path)
                    self.file_hashes[file_path] = self.calculate_file_hash(file_path)
                    
                    # 更新列表显示
                    self.message_queue.put(("update_list", os.path.basename(file_path)))
                    added_count += 1
                else:
                    failed_count += 1
            
            # 合并向量库
            if added_count > 0:
                self.merge_vectorstores()
            
            # 保存哈希记录
            self.save_hashes()
            
            # 准备结果消息
            result_msg = []
            if added_count > 0:
                result_msg.append(f"成功添加 {added_count} 个PDF文件")
            if skipped_count > 0:
                result_msg.append(f"跳过 {skipped_count} 个已存在文件")
            if failed_count > 0:
                result_msg.append(f"处理失败 {failed_count} 个文件")
            
            # 更新界面状态
            self.message_queue.put(("message", "系统", "\n".join(result_msg)))
            self.message_queue.put(("update_status", f"完成添加PDF文件"))
            self.message_queue.put(("enable_button", "add_btn"))
            
            # 更新系统就绪消息
            if added_count > 0:
                self.update_system_ready_message()
        
        except Exception as e:
            error_msg = f"添加PDF文件时出错: {str(e)}"
            print(error_msg)
            self.message_queue.put(("error", error_msg))
            self.message_queue.put(("enable_button", "add_btn"))

    def remove_pdfs(self):
        """从系统中移除选定的PDF文件
        
        1. 获取列表框中选中的文件
        2. 从各数据结构中移除对应项
        3. 重新合并剩余向量库
        4. 更新磁盘存储
        5. 刷新界面状态
        """
        # 获取选中的文件索引（按从后往前顺序处理）
        selected_indices = list(self.pdf_list.curselection())
        if not selected_indices:
            self.message_queue.put("系统", "请先选择要删除的PDF文件")
            return
        
        # 确认对话框
        if not messagebox.askyesno(
            "确认删除",
            f"确定要删除选中的 {len(selected_indices)} 个PDF文件吗？\n"
            "此操作将同时移除相关的向量数据且不可恢复！"
        ):
            return
        
        # 禁用按钮防止重复操作
        self.remove_btn.config(state=tk.DISABLED)
        self.update_status(f"正在移除 {len(selected_indices)} 个PDF文件...")
        
        # 在新线程中执行移除操作
        Thread(target=self._process_removal, args=(selected_indices,), daemon=True).start()

    def _process_removal(self, selected_indices):
        """实际执行移除操作的线程方法"""
        try:
            removed_files = []
            
            # 按从后往前顺序处理（避免索引变化问题）
            for i in reversed(selected_indices):
                if i < len(self.pdf_paths):
                    file_path = self.pdf_paths[i]
                    
                    # 从各数据结构中移除
                    removed_files.append(os.path.basename(file_path))
                    del self.pdf_paths[i]
                    if file_path in self.file_hashes:
                        del self.file_hashes[file_path]
                    if file_path in self.individual_vectorstores:
                        del self.individual_vectorstores[file_path]
                    
                    # 从列表框中移除
                    self.message_queue.put(("remove_from_list", i))
            
            # 如果有文件被移除
            if removed_files:
                # 重新合并剩余向量库
                self.merge_vectorstores()
                
                # 保存更新后的数据
                self.save_hashes()
                self.save_vectorstore(self.vectorstore_path)
                
                # 准备结果消息
                result_msg = (
                    f"已成功移除 {len(removed_files)} 个PDF文件:\n"
                    + "\n".join(f"• {name}" for name in removed_files)
                )
                
                # 更新系统状态
                self.message_queue.put(("message", "系统", result_msg))
                self.update_system_ready_message()
            else:
                self.message_queue.put(("message", "系统", "没有文件被移除"))
        
        except Exception as e:
            error_msg = f"移除PDF文件时出错: {str(e)}"
            self.message_queue.put(("error", error_msg))
        
        finally:
            # 恢复按钮状态
            self.message_queue.put(("enable_button", "remove_btn"))
            self.message_queue.put(("update_status", "准备就绪"))
    

    def select_all_pdfs(self):
        """选择列表框中的所有PDF文件"""
        #self.pdf_list.selection_clear(0, tk.END)  # 先清除现有选择
        self.pdf_list.selection_set(0, tk.END)    # 选择所有项目
        
        # 可选：滚动到列表底部确保所有选项可见
        if self.pdf_list.size() > 0:
            self.pdf_list.see(tk.END)
        
        # 更新状态显示（可选）
        selected_count = len(self.pdf_list.curselection())
        self.update_status(f"已选择 {selected_count} 个PDF文件")

    def select_non_pdfs(self):
        """取消选择列表框中的PDF文件"""
        self.pdf_list.selection_clear(0, tk.END)  # 先清除现有选择
        # 更新状态显示（可选）
        selected_count = 0
        self.update_status(f"已取消选择 PDF文件")
        
    def load_existing_data(self):
        self.send_btn.config(state=tk.DISABLED)
        print("加载已有数据...")
        """加载已有数据"""
        # 加载设置
        self.load_settings()
        
        # 尝试加载向量库
        print("寻找已有向量数据库" + self.vectorstore_path)
        if os.path.exists(self.vectorstore_path):
            print("正在加载已有向量数据库...")
            self.message_queue.put(("message", "系统", "正在加载已有向量数据库..."))
            if self.load_vectorstore(self.vectorstore_path):
                print("向量数据库加载成功...")
                self.message_queue.put(("message", "系统", "向量数据库加载成功"))
                self.load_pdf_list()
                self.update_system_ready_message()
            else:
                print("向量数据库加载失败...")
                self.message_queue.put(("message", "系统", "向量数据库加载失败"))
        else:
            print("未找到已有数据...")
            self.message_queue.put(("status", "准备就绪，请导入PDF文件"))
    
    def process_messages(self):
        try:
            while True:
                try:
                    msg_type, *content = self.message_queue.get_nowait()
                    
                    if msg_type == "status":
                        self.update_status(content[0])
                    elif msg_type == "message":
                        self.message_queue.put(content[0], content[1])
                    elif msg_type == "error":
                        self.message_queue.put("错误", content[0])
                    elif msg_type == "enable_buttons":
                        self.import_btn.config(state=tk.NORMAL)
                        self.browse_btn.config(state=tk.NORMAL)
                        self.send_btn.config(state=tk.NORMAL)
                    #add pdf
                    elif msg_type == "update_list":
                        self.pdf_list.insert(tk.END, content[0])
                    elif msg_type == "enable_button":
                        if content[0] == "add_btn":
                            self.add_btn.config(state=tk.NORMAL)
                    #delete pdf
                    elif msg_type == "remove_from_list":
                        self.pdf_list.delete(content[0])
                    elif msg_type == "enable_button":
                        if content[0] == "remove_btn":
                            self.remove_btn.config(state=tk.NORMAL)
                except queue.Empty:
                    break
        finally:
            self.root.after(100, self.process_messages)
    
    def update_status(self, message):
        self.status_var.set(message)
    
    def process_single_pdf(self, pdf_path):
        """处理单个PDF文件并创建其单独的向量库
        
        Args:
            pdf_path (str): PDF文件的完整路径
            
        Returns:
            bool: 处理是否成功
        """
        try:
            # 计算文件哈希
            file_hash = self.calculate_file_hash(pdf_path)
            
            # 检查是否已经处理过且未修改
            if pdf_path in self.file_hashes and self.file_hashes[pdf_path] == file_hash:
                if pdf_path in self.individual_vectorstores:
                    print(f"文件未修改，使用缓存: {pdf_path}")
                    return True
                # 否则需要重新处理
                
            # 读取PDF内容
            reader = PdfReader(pdf_path)
            documents = []
            for  page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": pdf_path, "page": page_num}
                    ))
                    print (page_num)
            
            if not documents:
                self.failed_files[pdf_path] = "无法提取文本内容"
                print(f"无法提取文本内容: {pdf_path}")
                return False
            
            # 分割文本
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            # chunks = text_splitter.split_text("\n".join(file_text))
            split_docs = text_splitter.split_documents(documents)

            
            if not split_docs:
                self.failed_files[pdf_path] = "文本分割后无有效内容"
                print(f"文本分割后无有效内容: {pdf_path}")
                return False
            
            # 创建嵌入模型
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': self.get_device()},
                encode_kwargs={
                    'batch_size': 32,
                    'device': self.get_device(),
                    'normalize_embeddings': True
                }
            )
            
            # 为当前PDF创建单独的向量库
            # vectorstore = FAISS.from_texts(chunks, embeddings)
            vectorstore = FAISS.from_documents(
                split_docs,  # 使用Document对象而非纯文本
                embeddings
            )
            # 保存到单独向量库字典
            self.individual_vectorstores[pdf_path] = vectorstore
            
            # 更新文件哈希记录
            self.file_hashes[pdf_path] = file_hash
            
            print(f"成功处理PDF文件: {pdf_path} (生成 {len(documents)} 个文本块)")
            return True
            
        except PdfReadError as e:
            self.failed_files[pdf_path] = "PDF文件损坏或加密"
            print(f"PDF读取错误: {pdf_path} - {str(e)}")
            return False
        except Exception as e:
            self.failed_files[pdf_path] = str(e)
            print(f"处理PDF文件出错: {pdf_path} - {str(e)}")
            return False

    def merge_vectorstores(self):
        """合并所有单独的向量库到主向量库"""
        if not self.individual_vectorstores:
            self.vectorstore = None
            self.message_queue.put(("message", "系统", "没有可用的向量库"))
            return False
        
        try:
            print("开始合并向量库...")
            start_time = datetime.now()
            
            # 获取第一个向量库作为基础
            first_path = next(iter(self.individual_vectorstores))
            self.vectorstore = self.individual_vectorstores[first_path]
            
            # 合并剩余的向量库
            merge_count = 0
            for path, vectorstore in self.individual_vectorstores.items():
                if path != first_path:
                    self.vectorstore.merge_from(vectorstore)
                    merge_count += 1
            
            # 保存合并后的向量库
            self.save_vectorstore(self.vectorstore_path)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"向量库合并完成，共合并 {merge_count + 1} 个向量库，耗时 {elapsed:.2f} 秒")
            
            # 更新系统信息
            self.message_queue.put(("message", "系统", 
                                  f"向量库合并完成，共包含 {len(self.individual_vectorstores)} 个PDF文档"))
            return True
            
        except Exception as e:
            error_msg = f"合并向量库时出错: {str(e)}"
            print(error_msg)
            self.message_queue.put(("error", error_msg))
            return False

    def load_pdf_list(self):
        """加载已有PDF列表到管理界面"""
        print("加载已有PDF列表到管理界面")
        self.pdf_list.delete(0, tk.END)
        self.pdf_paths = []
        
        # 从文件哈希记录中加载PDF列表
        valid_pdfs = []
        for pdf_path in self.file_hashes.keys():
            if os.path.exists(pdf_path):
                valid_pdfs.append(pdf_path)
            else:
                print(f"文件不存在，已移除: {pdf_path}")
                del self.file_hashes[pdf_path]
        
        # 更新文件哈希记录
        self.save_hashes()
        
        # 检查哪些PDF已有向量库
        loaded_count = 0
        for pdf_path in valid_pdfs:
            filename = os.path.basename(pdf_path)
            self.pdf_list.insert(tk.END, filename)
            self.pdf_paths.append(pdf_path)
            
            # 检查是否已有向量库
            if pdf_path in self.individual_vectorstores:
                loaded_count += 1
        
        # 更新系统信息
        self.pdf_count_info.set(f"已加载PDF: {len(valid_pdfs)}")
        self.vectorstore_info.set(f"向量库: 已加载 ({loaded_count}/{len(valid_pdfs)})")
        
        if valid_pdfs:
            print(f"已加载 {len(valid_pdfs)} 个PDF文件到列表")
        else:
            print("没有可用的PDF文件")
            
    def create_widgets(self):
        """使用Grid实现纵向三块 + 中部左右布局"""
        self.root.grid_rowconfigure(1, weight=1)  # 中间区域可扩展
        self.root.grid_columnconfigure(0, weight=1)

        # 顶部状态栏
        self._create_top_frame()

        # 中部左右主区域
        self._create_main_frame()

        # 底部输入栏
        self._create_bottom_frame()


    def _create_top_frame(self):
        """创建顶部状态栏"""
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.grid(row=0, column=0, sticky="ew")

        self.root.grid_columnconfigure(0, weight=1)

        # 状态标签
        self.status_var = tk.StringVar(value="准备就绪")
        ttk.Label(top_frame, textvariable=self.status_var).pack(side=tk.LEFT, expand=True)

        # 模型信息标签
        self.model_info_var = tk.StringVar()
        ttk.Label(top_frame, textvariable=self.model_info_var).pack(side=tk.RIGHT)


    def _create_main_frame(self):
        """主区域：左右布局"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=1, column=0, sticky="nsew")

        main_frame.grid_columnconfigure(0, weight=3)  # 左70%
        main_frame.grid_columnconfigure(1, weight=1)  # 右30%
        main_frame.grid_rowconfigure(0, weight=1)

        # 左侧对话区域
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # 使用ScrolledText替代HTMLLabel以支持文本选择和复制
        self.conversation = scrolledtext.ScrolledText(
            left_panel,
            wrap=tk.WORD,
            width=80,
            height=25,
            font=('Arial', 11),
            state='normal'
        )
        self.conversation.pack(fill=tk.BOTH, expand=True)

        clear_btn = ttk.Button(
            left_panel,
            text="清空对话",
            command=self.clear_conversation,
            width=15
        )
        clear_btn.pack(pady=5)

        # 右侧控制区域
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky="nsew")

        self._create_pdf_notebook(right_panel)
        self._create_model_settings(right_panel)

    def _create_bottom_frame(self):
        """底部输入区域"""
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.grid(row=3, column=0, sticky="ew")
        bottom_frame.grid_columnconfigure(0, weight=3)  # 左70%
        bottom_frame.grid_columnconfigure(1, weight=1)  # 右30%
        
        # 左侧：用户输入区域
        left_input_frame = ttk.Frame(bottom_frame)
        left_input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        self.user_input = scrolledtext.ScrolledText(
            left_input_frame,
            wrap=tk.WORD,
            width=80,
            height=5
        )
        self.user_input.pack(side=tk.LEFT,fill=tk.BOTH, expand=True)
        # 发送按钮
        self.send_btn = ttk.Button(
            left_input_frame,
            text="发送",
            command=self.send_message,
            width=5
        )
        self.send_btn.pack(side=tk.RIGHT,fill=tk.Y)
        
        # 绑定回车键
        self.user_input.bind("<Return>", lambda e: self.send_message())
        
        # 右侧：控制区域
        right_control_frame = ttk.Frame(bottom_frame)
        right_control_frame.grid(row=0, column=1, sticky="nsew")
        self._create_system_info(right_control_frame)


        
    def _create_pdf_notebook(self, parent):
        """创建PDF管理笔记本"""
        self.pdf_notebook = ttk.Notebook(parent)
        self.pdf_notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 第一页：PDF文件管理
        self.pdf_manage_frame = ttk.Frame(self.pdf_notebook)
        self.pdf_notebook.add(self.pdf_manage_frame, text="PDF文件管理")
        
        # 列表控件
        self.pdf_list = tk.Listbox(
            self.pdf_manage_frame,
            selectmode=tk.EXTENDED,
            height=15,
            exportselection=False
        )
        self.pdf_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.pdf_list.bind("<Double-Button-1>", self.open_selected_pdf)
        self.pdf_list.bind("<<ListboxSelect>>", self.show_selected_pdf_info)

        
        # 按钮框架
        btn_frame = ttk.Frame(self.pdf_manage_frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        # 添加按钮
        self.add_btn = ttk.Button(
            btn_frame,
            text="添加",
            command=self.add_pdfs,
            width=8
        )
        self.add_btn.pack(side=tk.LEFT, padx=2)
        
        # 删除按钮
        self.remove_btn = ttk.Button(
            btn_frame,
            text="删除",
            command=self.remove_pdfs,
            width=8
        )
        self.remove_btn.pack(side=tk.LEFT, padx=2)
        
        # 全部选择按钮
        self.select_all_btn = ttk.Button(
            btn_frame,
            text="全选",
            command=self.select_all_pdfs,
            width=8
        )
        self.select_all_btn.pack(side=tk.LEFT, padx=2)
        
        # 取消选择按钮
        self.select_non_btn = ttk.Button(
            btn_frame,
            text="取消选择",
            command=self.select_non_pdfs,
            width=8
        )
        self.select_non_btn.pack(side=tk.LEFT, padx=2)
        
        # 第二页：PDF批量导入
        self.batch_import_frame = ttk.Frame(self.pdf_notebook)
        self.pdf_notebook.add(self.batch_import_frame, text="PDF批量导入")
        
        # PDF路径输入框和浏览按钮
        ttk.Label(self.batch_import_frame, text="PDF文件夹:").pack(anchor=tk.W, pady=(5, 0))
        self.pdf_folder_var = tk.StringVar()
        self.pdf_entry = ttk.Entry(self.batch_import_frame, textvariable=self.pdf_folder_var, width=30)
        self.pdf_entry.pack(fill=tk.X, padx=5, pady=5)
        self.browse_btn = ttk.Button(
            self.batch_import_frame, 
            text="浏览...", 
            command=self.browse_folder
        )
        self.browse_btn.pack(pady=5)
        
        # 导入按钮
        self.import_btn = ttk.Button(
            self.batch_import_frame,
            text="导入PDF",
            command=self.import_pdfs,
            width=15
        )
        self.import_btn.pack(pady=10)
        
        # 强制重新处理按钮
        self.reprocess_btn = ttk.Button(
            self.batch_import_frame,
            text="强制重新处理",
            command=self.force_reprocess,
            width=15
        )
        self.reprocess_btn.pack(pady=5)
        
    def get_pdf_preview(self, filepath, count = 100):
        """提取PDF文件的前100个字符作为预览内容"""
        preview_text = ""
        try:
            with open(filepath, 'rb') as f:
                reader = PdfReader(f)
                first_page_text = reader.pages[0].extract_text()
                if first_page_text:
                    preview_text = first_page_text[:count].replace('\n', ' ') + "..."
        except Exception as e:
            preview_text = f"无法读取文件内容: {str(e)}"
        return preview_text

    def show_selected_pdf_info(self, event):
        """显示选中PDF的文件信息（调用get_pdf_preview获取内容预览）"""
        selected_indices = self.pdf_list.curselection()
        
        if not selected_indices:
            return
        
        # 只处理第一个选中的文件
        selected_index = selected_indices[0]
        
        if selected_index < len(self.pdf_paths):
            file_path = self.pdf_paths[selected_index]
            
            # 调用独立的提取函数获取预览内容
            preview_text = self.get_pdf_preview(file_path,100)
            
            # 显示文件信息
            file_name = os.path.basename(file_path)
            info_text = f"选中的文件：{file_name}\n路径：{file_path}\n内容预览：{preview_text}"
            self.status_var.set(info_text)
            
    def open_selected_pdf(self, event):
        """双击打开选中的PDF文件"""
        # 获取选中的索引
        selected_indices = self.pdf_list.curselection()
        
        if not selected_indices:
            return
        
        # 只处理第一个选中的文件（双击通常只选中一个）
        selected_index = selected_indices[0]
        
        # 检查索引是否有效
        if selected_index < len(self.pdf_paths):
            file_path = self.pdf_paths[selected_index]
            
            try:
                # 使用系统默认程序打开PDF
                if os.name == 'nt':  # Windows
                    os.startfile(file_path)
                elif os.name == 'posix':  # macOS/Linux
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', file_path], check=True)
                    else:  # Linux
                        subprocess.run(['xdg-open', file_path], check=True)
            except Exception as e:
                self.message_queue.put("错误", f"无法打开PDF文件: {str(e)}")

    def update_temp_slider(self, event=None):
        try:
            temp = float(self.temp_var.get())
            if temp < 0 or temp > 1:
                raise ValueError
            self.temp_slider.set(temp)
        except ValueError:
            messagebox.showerror("错误", "请输入0到1之间的数字")
            self.temp_var.set(0.7)
            self.temp_slider.set(0.7)
        
    def browse_folder(self):
        folder = filedialog.askdirectory(title="选择包含PDF的文件夹")
        if folder:
            self.pdf_folder_var.set(folder)
    
    def update_temp_value(self, val):
        self.temp_var.set(round(float(val), 2))
        
    
    def get_conversation_summary(self, history):
        """
        获取对话总结的优化版本
        """
        print("总结一下历史记录")
        if not isinstance(history, list) or len(history) <= 1:
            print("没有历史记录")
            return ""
        
        total_rounds = len(history)
        
        # 获取最近2轮对话
        recent_actual = [str(item) for item in history[-2:]]
        
        # 定义总结范围：第3轮到第10轮（0-based索引）
        SUMMARY_START = 3
        SUMMARY_END = 10
        
        # 检查是否有足够的历史记录进行总结
        if total_rounds <= SUMMARY_START:
            print("历史记录<3")
            return "最近对话：\n" + "\n".join(recent_actual)
        
        # 获取需要总结的对话段
        to_summarize = [str(item) for item in history[SUMMARY_START:min(10, total_rounds)]]
        
        if not to_summarize:
            print("历史记录<3被发回")
            return "最近对话：\n" + "\n".join(recent_actual)
        
        # 生成总结...
        summary_prompt = f"""请用简洁的1-2句话总结以下较早的对话历史：
                            {chr(10).join(to_summarize)}
                            总结："""
       
        try:
            summary = self.llmClient(
                baseUrl=self.config_var.get(),
                backend=self.backend_var.get(),
                model=self.model_var.get(),
                temperature=self.temp_var.get()
            ).generate(prompts=[summary_prompt]).generations[0][0].text
            print("历史记录3~10被总结")
            return f"""较早对话总结：{summary}
                        最近对话记录：
                        {chr(10).join(recent_actual)}"""
            
        except Exception as e:
            print(f"生成总结时出错：{e}")
            return "最近对话：\n" + "\n".join(recent_actual) 
            
    def send_message(self, event=None):
        #没有向量库
        if not hasattr(self, 'vectorstore') or self.vectorstore is None:
            self.message_queue.put("message", "请先成功导入PDF文件或加载向量库")
            return

        user_text = self.user_input.get("1.0", tk.END).strip()
        #没有用户提示就不发送
        if not user_text:
            return
        #没有选定模型也不发送
        print(self.model_var.get())
        print(self.config_var.get())
        if not self.model_var.get().strip():
            self.message_queue.put("message", "请先选定模型")
            return
        if not hasattr(self, 'qa_chain') or self.qa_chain is None:
            try:
                # 获取用户设置的系统提示
                system_prompt = self.prompt_input.get("1.0", tk.END)
                if system_prompt:
                    # 如果有自定义系统提示，使用增强版本
                    prompt_template = f"""
                                                {system_prompt}

                                                You are answering based on the provided context. Please follow these additional guidelines:
                                                - Base your answers strictly on the provided context, Focus on the AUTHOR'S OWN TEXT, not citations or the paper's references
                                                - Reference previous conversation when relevant, but prioritize current context  
                                                - Quote specific evidence and explain your reasoning
                                                - Express confidence levels based on available information
                                                - If information is missing, state this clearly

                                                Context: {{context}}

                                                Question: {{question}}

                                                Answer:"""
                else:
                    # 默认 RAG prompt
                    prompt_template = """
                                                You are an expert AI assistant that provides accurate answers based on provided context while maintaining natural conversation flow.

                                                Guidelines:
                                                - Answer based strictly on the provided context, Focus on the AUTHOR'S OWN TEXT, not citations or paper's references
                                                - Reference previous conversation when relevant, but prioritize current context
                                                - Quote specific evidence and explain your reasoning  
                                                - Express confidence levels (High/Medium/Low) based on available information
                                                - If information is missing, state this clearly and offer to help refine the question

                                                Context: {context}

                                                Question: {question}

                                                Answer:"""
                     
                
                PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )                
                
                # 如果没有QA链则创建新的QA链
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llmClient(
                        baseUrl=self.config_var.get(),
                        backend=self.backend_var.get(),
                        model=self.model_var.get(),
                        temperature=self.temp_var.get(),
                    ),
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}), #其实这样可能会导致文档间的不均衡
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT},
                    verbose=True
                )
            except Exception as e:
                print(f"设置QA链时出错: {str(e)}")

        self.user_input.delete("1.0", tk.END)
        self.send_btn.config(state=tk.DISABLED)      
        Thread(target=self.get_answer, args=(user_text,), daemon=True).start()
    
    def get_answer(self, question):
        try:
            if self.qa_chain is None:
                return
            target_file = ""
            # 构建完整的查询内容
            query_parts = []
            # 构建增强问题指向相应文件
            print("try target some pdf")
            selected_indices = self.pdf_list.curselection()
            if selected_indices:
                target_file = [self.get_pdf_preview(self.pdf_paths[i],150) for i in selected_indices]
                print(target_file)
                query_parts.append(f"Priority Documents: Please prioritize information from these documents: {target_file}")
            # 添加用户问题
            print("show questions")
            query_parts.append(f"User Question: {question}")
            # 组合完整查询
            enhanced_query = "\n\n".join(query_parts)
            
            print(enhanced_query)
            # 更新对话记录（只显示原始问题）
            self.update_conversation("User", question)
            # 调用 QA 链
            result = self.qa_chain({"query": enhanced_query})
            query_answer = result["result"]
            # 构建对话问题
            question_parts = []
            # 添加RAG结果
            question_parts.append(f"!Important: RAG search result:{re.sub(r'<think>.*?</think>', '', query_answer, flags=re.DOTALL)}");
            # 添加用户原始提问
            question_parts.append(f"User Question: {question}")         
            # 包含对话历史
            print("try conversation_summary")
            conversation_summary  = self.get_conversation_summary(self.conversation_history)
            print(conversation_summary)
            if conversation_summary:
                print("add Hx")
                question_parts.append(f"PS: Conversation history summary: {conversation_summary}")            
            # 组合完整提问
            enhanced_question = "\n\n".join(question_parts) 
            
            try:
                system_prompt = self.prompt_input.get("1.0", tk.END)
                #messages = [
                #           {"role": "system", "content": system_prompt},
                #            {"role": "user", "content": enhanced_question}
                #       ]
                messages =f"{system_prompt}\n\n User:{enhanced_question} "

                answer = self.llmClient(
                    baseUrl=self.config_var.get(),
                    backend=self.backend_var.get(),
                    model=self.model_var.get(),
                    temperature=self.temp_var.get()
                ).generate(prompts=[messages]).generations[0][0].text
                    # 处理来源信息
                print("show sources")
                sources = []
                for doc in result["source_documents"]:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        source = f"[{os.path.basename(doc.metadata['source'])}]({doc.metadata['source']})\n"
                        if source not in sources:
                            sources.append(source)
                # 构建响应
                response = f"回答: {answer}"
                if sources:
                    response += f"\n\n来源: {', '.join(sources)}"

                self.update_conversation("AI", response)                 
                
            except Exception as e:
                print(f"生成总结时出错：{e}")
               
        except Exception as e:
            print(f"获取答案时出错: {str(e)}")
            self.message_queue.put(("error", f"获取答案时出错: {str(e)}"))
        finally:
            self.message_queue.put(("enable_buttons", True))
    # 回调函数
    def update_model_list(self, event=None):
        mbackend = self.backend_dropdown.get() 
        self.ollama_models = self.get_available_models(mbackend)
        self.message_queue.put("模型","self.ollama_models")
        print (self.ollama_models)
        # 更新模型列表
        self.model_dropdown['values'] = self.ollama_models
        self.config_entry['state'] = 'normal'
        if mbackend == "ollama":
            self.config_var.set("http://localhost:11434")
        elif mbackend == "lmstudio":
            self.config_var.set("http://localhost:1234")
        elif mbackend == "openai":
            self.config_var.set("输入API Key")  # 清空用于输入API Key
        
        # 默认选择第一个模型
        if self.model_dropdown['values']:
            self.model_var.set(self.model_dropdown['values'][0])
 
    def update_model(self, event):
        self.update_status(f"已选择模型: {self.model_dropdown.get()}")
    def on_model_change(self, event):
        self.update_status(f"已选择后端: {self.backend_dropdown.get()}")
    
    def _create_model_settings(self, parent):
        """创建带标签页的模型和提示设置区域"""
        # 创建Notebook（标签页容器）
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 第一个标签页：模型设置
        model_tab = ttk.Frame(notebook, padding="10")
        notebook.add(model_tab, text="模型设置")
        
        # 模型选择框架
        backend_frame = ttk.Frame(model_tab)
        backend_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(backend_frame, text="后端类型:").pack(side=tk.LEFT)
        self.backend_var = tk.StringVar(value="ollama")
        self.backend_dropdown = ttk.Combobox(
            backend_frame,
            textvariable=self.backend_var,
            values=["ollama", "lmstudio"]#, "openai"],
            state="readonly"
        )
        self.backend_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.backend_dropdown.bind("<<ComboboxSelected>>", self.update_model_list)

        # 模型选择
        models_frame = ttk.Frame(model_tab)
        models_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(models_frame, text="选择模型:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(
            models_frame,
            textvariable=self.model_var,
            state="readonly"
        )
        self.model_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.model_dropdown.bind("<<ComboboxSelected>>", self.update_model)

        # API Key/网络配置
        config_frame = ttk.Frame(model_tab)
        config_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(config_frame, text="配置:").pack(side=tk.LEFT)
        self.config_var = tk.StringVar()
        self.config_entry = ttk.Entry(
            config_frame,
            textvariable=self.config_var
        )
        self.config_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 温度参数控制
        temp_frame = ttk.Frame(model_tab)
        temp_frame.pack(fill=tk.X)

        ttk.Label(temp_frame, text="温度(0-1):").pack(side=tk.LEFT, pady=(5, 0))
        self.temp_entry = ttk.Entry(temp_frame, textvariable=self.temp_var, width=5)
        self.temp_entry.pack(side=tk.LEFT, padx=5, pady=(5, 0))
        self.temp_entry.bind("<Return>", self.update_temp_slider)

        self.temp_slider = ttk.Scale(
            model_tab, 
            from_=0, 
            to=1, 
            variable=self.temp_var,
            command=self.update_temp_value
        )
        self.temp_slider.pack(fill=tk.X, pady=5)
        
        # 最大token数
        max_tokens_frame = ttk.Frame(model_tab)
        max_tokens_frame.pack(fill=tk.X)

        ttk.Label(max_tokens_frame, text="最大token数:").pack(side=tk.LEFT, pady=(5, 0))
        self.max_tokens_entry = ttk.Entry(max_tokens_frame, textvariable=self.max_tokens_var, width=5)
        self.max_tokens_entry.pack(side=tk.LEFT, padx=5, pady=(5, 0))
        self.max_tokens_entry.bind("<Return>", lambda e: self.max_tokens_slider.set(self.max_tokens_var.get()))

        self.max_tokens_slider = ttk.Scale(
            model_tab,
            from_=512,
            to=4000,
            variable=self.max_tokens_var,
            command=lambda v: self.max_tokens_var.set(int(float(v)))
        )
        self.max_tokens_slider.pack(fill=tk.X, pady=5)
        
        # 第二个标签页：系统提示
        prompt_tab = ttk.Frame(notebook, padding="10")
        notebook.add(prompt_tab, text="系统提示词")
        
        self.prompt_input = scrolledtext.ScrolledText(
            prompt_tab,
            wrap=tk.WORD,
            width=25,
            height=3,
            font=("Arial", 9)
        )
        self.prompt_input.pack(fill=tk.BOTH, expand=True)
        self.prompt_input.insert(tk.END, self.default_prompt)
        # 应用提示词
        self.apply_prompt_btn = ttk.Button(
            prompt_tab,
            text="应用",
            command=self.apply_prompt,
            width=8
        )
        self.apply_prompt_btn.pack(side=tk.RIGHT, padx=2)
        
    def get_available_models(self,backend):
        return self.client.get_available_models(backend)
    def llmClient(self,baseUrl,backend,model,temperature):
        return self.client.llmClient(backend, baseUrl,model)
    def apply_prompt(self):
        print("重置QA chain")
        self.qa_chain = None
    def _create_system_info(self, parent):
        """创建系统信息区域"""
        info_frame = ttk.LabelFrame(parent, text="系统信息", padding="10")
        info_frame.pack(fill=tk.X)
        
        # 向量库信息
        self.vectorstore_info = tk.StringVar(value="向量库: 未加载")
        ttk.Label(info_frame, textvariable=self.vectorstore_info).pack(anchor=tk.W)
        
        # PDF数量信息
        self.pdf_count_info = tk.StringVar(value="已加载PDF: 0")
        ttk.Label(info_frame, textvariable=self.pdf_count_info).pack(anchor=tk.W)
        
        # 最后更新时间
        self.last_update_info = tk.StringVar(value="最后更新: 无")
        ttk.Label(info_frame, textvariable=self.last_update_info).pack(anchor=tk.W)
        
        # 保存设置按钮
        save_btn = ttk.Button(
            info_frame,
            text="保存当前设置",
            command=self.save_settings,
            width=15
        )
        save_btn.pack(pady=5)
        

    def load_hashes(self):
        try:
            if os.path.exists(self.hash_db_path):
                with open(self.hash_db_path, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}

    def save_hashes(self):
        with open(self.hash_db_path, 'w') as f:
            json.dump(self.file_hashes, f)

    def calculate_file_hash(self, filepath):
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def need_reprocess(self, folder):
        current_files = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith('.pdf'):
                    full_path = os.path.join(root, file)
                    current_files.append(full_path)

        need_process = False
        
        if set(self.file_hashes.keys()) - set(current_files):
            need_process = True
        for file in current_files:
            current_hash = self.calculate_file_hash(file)
            if file not in self.file_hashes or self.file_hashes[file] != current_hash:
                need_process = True
                self.file_hashes[file] = current_hash
        if need_process:
            self.save_hashes()
        return need_process

    def get_local_ollama_models(self):
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                models = [line.split()[0] for line in lines[1:] if line.strip()]
                return models if models else ["llama2"]
            return ["llama2"]
        except:
            return ["llama2"]
    
    def import_pdfs(self):
        folder = self.pdf_folder_var.get()
        if not folder:
            self.update_status("请先选择PDF文件夹")
            return
        
        # 清空列表
        self.pdf_paths = []
        self.pdf_list.delete(0, tk.END)
        self.success_files = []
        self.failed_files = {}
        
        # 禁用按钮防止重复点击
        self.import_btn.config(state=tk.DISABLED)
        self.browse_btn.config(state=tk.DISABLED)
        self.update_status("正在扫描PDF文件...")
        
        # 在新线程中处理PDF
        Thread(target=self.process_pdfs, args=(folder,), daemon=True).start()
    
    def record_pdf_hashes(self):
        with open(self.hash_db_path, 'w') as f:
            json.dump(self.file_hashes, f)
    
    def process_pdfs(self, folder):
        try:
            # 递归扫描所有PDF文件
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        full_path = os.path.join(root, file)
                        if full_path not in self.pdf_paths:
                            self.pdf_paths.append(full_path)
                            self.pdf_list.insert(tk.END, file)
                            self.file_hashes[full_path] = self.calculate_file_hash(full_path)
            
            if not self.pdf_paths:
                self.message_queue.put(("status", "未找到PDF文件"))
                return
            
            self.message_queue.put(("message", f"找到 {len(self.pdf_paths)} 个PDF文件，正在处理..."))
            
            
            # 分割文本
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            
                
                
            # 读取和分割PDF文本
            documents = []
           # print(f"[调试] 开始处理 {len(self.pdf_paths)} 个PDF文件")  # 调试语句
            for pdf_path in self.pdf_paths:
                print(f"\n 正在处理: {os.path.basename(pdf_path)}:{pdf_path}")  # 显示当前文件名
                try:
                    reader = PdfReader(pdf_path)
                    file_doc = []
                   # print(f"[调试] 总页数: {len(reader.pages)}")  # 显示PDF页数
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            file_doc.append(Document(
                                page_content=text,
                                metadata={"source": pdf_path, "page": page_num}
                            ))
                           # file_doc.append(text)
                    if file_doc:
                        split_docs = text_splitter.split_documents(file_doc)
                        if not split_docs:
                            self.message_queue.put(("error", "文件分割后无有效内容"))
                            return
                        documents.extend(split_docs)
                        self.success_files.append(pdf_path)
                        print("成功！")
                    else:
                        self.failed_files[pdf_path] = "无法提取文本内容"
                except PdfReadError as e:
                    self.failed_files[pdf_path] = "PDF文件损坏或加密"
                except Exception as e:
                    self.failed_files[pdf_path] = str(e)
                    print(f"处理{os.path.basename(pdf_path)}出错: {str(e)}")

            
            # 在对话中显示处理结果
            if self.success_files:
                self.message_queue.put("status", f"成功导入 {len(self.success_files)} 个PDF文件")
            if self.failed_files:
                self.message_queue.put("status", f"导入失败 {len(self.failed_files)} 个PDF文件")
                for file, reason in self.failed_files.items():
                    self.message_queue.put("status", f"失败文件: {os.path.basename(file)} - 原因: {reason}")
            print("创建向量存储")
            # 创建向量存储
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    model_kwargs={'device': self.get_device()},
                    encode_kwargs={
                        'batch_size': 32,
                        'device': self.get_device(),
                        'normalize_embeddings': True
                    }
                )
                
                # self.vectorstore = FAISS.from_texts(chunks, embeddings)
                self.vectorstore = FAISS.from_documents(documents, embeddings)
                self.save_vectorstore(self.vectorstore_path)
                self.record_pdf_hashes()
                success_msg = f"完成! 成功导入 {len(self.success_files)}/{len(self.pdf_paths)} 个PDF文件"
                self.message_queue.put(("status", success_msg))
                self.message_queue.put(("message", "系统", "RAG系统已准备好回答关于PDF内容的问题"))
                
            except Exception as e:
                error_msg = f"创建向量存储或QA链时出错: {str(e)}"
                self.message_queue.put(("error", error_msg))
                
        except Exception as e:
            error_msg = f"处理PDF时出错: {str(e)}"
            self.message_queue.put(("error", error_msg))
        finally:
            self.message_queue.put(("enable_buttons", True))
    
    
    def save_vectorstore(self, path="vectorstore"):
        print("保存向量数据库..."+path)
        try:
            if self.vectorstore:
                self.vectorstore.save_local(
                    folder_path=path,
                    index_name="index"
                )
                print("保存向量数据库成功！")
                return True
            return False
        except Exception as e:
            print(f"保存向量数据库失败: {str(e)}")
            return False
            
    def load_vectorstore(self, path="vectorstore"):
        print("加载向量数据库...")
        try:
            if not os.path.exists(path):
                return False
                
            device = self.device if hasattr(self, 'device') else 'cpu'
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': device}
            )
            
            self.vectorstore = FAISS.load_local(
                folder_path=path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            
            return True
            
        except Exception as e:
            print(f"加载向量数据库失败: {str(e)}")
            return False
                
    def handle_link_click(self, event):
        """处理对话中的链接点击"""
        widget = event.widget
        index = widget.index(f"@{event.x},{event.y}")
        
        # 获取点击位置的标签
        tags = widget.tag_names(index)
        
        for tag in tags:
            if tag.startswith("link_"):
                url = tag[5:]  # 去掉"link_"前缀
                webbrowser.open(url)
                break

    def clear_conversation(self):
        """清空对话历史"""
        self.conversation.config(state='normal')
        self.conversation.delete(1.0, tk.END)
        self.conversation.config(state='disabled')
        self.conversation_history = []
        self.message_queue.put("系统", "对话历史已清空")

    def update_system_ready_message(self):
        """更新系统就绪消息"""
        model = self.model_var.get()
        pdf_count = len(self.pdf_paths)
        self.model_info_var.set(f"当前模型: {model}")
        self.vectorstore_info.set(f"向量库: 已加载 ({len(self.individual_vectorstores)}个PDF)")
        self.pdf_count_info.set(f"已加载PDF: {pdf_count}")
        self.last_update_info.set(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        message = (
            f"RAG系统已准备好回答关于PDF内容的问题\n\n"
            f"当前模型: **{model}**\n"
            f"已加载PDF文档: **{pdf_count}**个\n"
            f"最后更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.message_queue.put("系统", message)
        self.send_btn.config(state=tk.NORMAL)  # 添加这行


    def save_settings(self):
        """保存当前设置"""
        settings = {
            "backend": self.backend_var.get(),
            "model": self.model_var.get(),
            "conf": self.config_var.get(),
            "temperature": self.temp_var.get(),
            "max_tokens": self.max_tokens_var.get(),
            "last_pdf_folder": self.pdf_folder_var.get(),
            "prompt": self.prompt_input.get("1.0", tk.END)
        }
        
        try:
            with open(self.settings_path, 'w') as f:
                json.dump(settings, f)
            self.message_queue.put("系统", "设置已保存")
        except Exception as e:
            self.message_queue.put("系统", f"保存设置失败: {str(e)}")

    def load_settings(self):
        """加载保存的设置"""
        print("加载设置...")

        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, 'r') as f:
                    settings = json.load(f)
                
                self.model_var.set(settings.get("model", self.ollama_models[0] if self.ollama_models else "llama2"))
                self.backend_var.set(settings.get("backend",'ollama'))
                self.config_var.set(settings.get("conf","http://localhost:11434"))
                self.temp_var.set(settings.get("temperature", 0.7))
                self.max_tokens_var.set(settings.get("max_tokens", 2000))
                self.pdf_folder_var.set(settings.get("last_pdf_folder", ""))
                self.prompt_input.set(settings.get("prompt", self.default_prompt))
                print("设置加载成功！")
                return True
        except Exception as e:
            print(f"加载设置失败: {str(e)}")
        return False

    def force_reprocess(self):
        """强制重新处理所有文件"""
        self.file_hashes = {}
        self.pdf_list.delete(0, tk.END)
        self.pdf_paths = []
        self.individual_vectorstores = {}
        self.vectorstore = None
        
        folder = self.pdf_folder_var.get()
        if folder:
            self.import_pdfs()
        else:
            self.message_queue.put("系统", "请先选择PDF文件夹")

    def update_conversation(self, sender, message):
        """更新对话显示，支持Markdown格式和超链接"""
        # 确保文本区域可编辑
        self.conversation.config(state='normal')
        
        # 添加发送者标签
        self.conversation.insert(tk.END, f"{sender}: ", "sender_tag")
        
        # 处理AI回复中的Markdown格式
        if sender == "AI":
            self.insert_formatted_text(message)
        else:
            self.conversation.insert(tk.END, message)

        
        # 添加分隔线
        self.conversation.insert(tk.END, "\n" + "-"*80 + "\n\n")
        
        # 配置样式
        self.configure_tags()
        
        # 禁止编辑但允许选择和复制
        self.conversation.config(state='disabled')
        
        # 滚动到底部
        self.conversation.see(tk.END)
        
        # 保存到历史
        self.conversation_history.append((sender, re.sub(r'<think>.*?</think>', '', message, flags=re.DOTALL)))
    
    def insert_formatted_text(self, message):
        """插入带有格式的文本"""
        pos = 0
        while pos < len(message):
            # 查找各种格式的开始标记
            bold_start = message.find("**", pos)
            italic_start = message.find("*", pos)
            code_start = message.find("```", pos)
            link_start = message.find("[", pos)
            
            # 找出最近的格式标记
            next_pos = min(
                x for x in [bold_start, italic_start, code_start, link_start, len(message)] 
                if x != -1
            )
            
            if next_pos == len(message):
                # 插入剩余普通文本
                self.conversation.insert(tk.END, message[pos:])
                break
            
            # 插入前面的普通文本
            if next_pos > pos:
                self.conversation.insert(tk.END, message[pos:next_pos])
            
            # 处理找到的格式
            if next_pos == bold_start:
                # 处理粗体
                end = message.find("**", next_pos + 2)
                if end == -1:
                    pos = next_pos + 2
                    continue
                bold_text = message[next_pos + 2:end]
                self.conversation.insert(tk.END, bold_text, "bold_tag")
                pos = end + 2
                
            elif next_pos == italic_start:
                # 处理斜体
                end = message.find("*", next_pos + 1)
                if end == -1:
                    pos = next_pos + 1
                    continue
                italic_text = message[next_pos + 1:end]
                self.conversation.insert(tk.END, italic_text, "italic_tag")
                pos = end + 1
                
            elif next_pos == code_start:
                # 处理代码块
                end = message.find("```", next_pos + 3)
                if end == -1:
                    pos = next_pos + 3
                    continue
                code_text = message[next_pos + 3:end]
                self.conversation.insert(tk.END, code_text, "code_tag")
                pos = end + 3
                
            elif next_pos == link_start:
                # 处理超链接
                end = message.find("]", next_pos)
                url_start = message.find("(", end)
                url_end = message.find(")", url_start)
                
                if -1 in (end, url_start, url_end):
                    pos = next_pos + 1
                    continue
                    
                link_text = message[next_pos + 1:end]
                url = message[url_start + 1:url_end]
                
                # 创建唯一标签名
                tag_name = f"link_{len(self.conversation.tag_names())}"
                
                # 插入链接文字并打标签
                self.conversation.insert(tk.END, link_text, tag_name)
                
                # 设置样式和绑定事件
                self.conversation.tag_config(tag_name, foreground="blue", underline=True)
                self.conversation.tag_bind(tag_name, "<Enter>", 
                                         lambda e: self.conversation.config(cursor="hand2"))
                self.conversation.tag_bind(tag_name, "<Leave>", 
                                         lambda e: self.conversation.config(cursor=""))
                self.conversation.tag_bind(tag_name, "<Button-1>", 
                                         lambda e, u=url: webbrowser.open(u))
                
                pos = url_end + 1
    
    def configure_tags(self):
        """配置各种文本格式的tag样式"""
        self.conversation.tag_config("sender_tag", font=('Noto Sans CJK SC', 11, 'bold'))
        self.conversation.tag_config("bold_tag", font=('Noto Sans CJK SC', 10, 'bold'))
        self.conversation.tag_config("italic_tag", font=('Noto Sans CJK SC', 10, 'italic'))
        self.conversation.tag_config("code_tag", 
                                   font=('Courier New', 10), 
                                   background='#f0f0f0',
                                   relief='raised',
                                   borderwidth=1)



if __name__ == "__main__":
    root = tk.Tk()
    app = PDFRAGApp(root)
    root.mainloop()