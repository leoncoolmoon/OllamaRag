import hashlib
import torch
import json
import os #用于保存文件打开文件之类的服务端操作
import webbrowser
import re
import requests
import json
import queue
from datetime import datetime
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.documents import Document  # 添加到文件顶部的导入部分
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Any, List, Optional, Literal
from pydantic import PrivateAttr
import argparse

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

class LangChainWrapper(LLM):
    _standalone_llm: StandaloneLLM = PrivateAttr()
    def __init__(self, standalone_llm: StandaloneLLM,**kwargs):
        super().__init__(**kwargs)  # 只传递父类需要的参数
        self._standalone_llm = standalone_llm

    
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
        return self._standalone_llm.generate(prompt, stop=stop or [], **kwargs)

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
    def llmClient(self, backend: str,
                  baseurl: str, model: str, temperature: float = 0.7,
                  api_key: Optional[str] = None, **kwargs):
        """创建LLM客户端实例"""
        try:
            if backend == 'ollama':
                return Ollama(
                    base_url=self.ensure_http_prefix(baseurl),
                    model=model,
                    temperature=temperature,
                    **kwargs
                )
            elif backend == 'lmstudio':
                # 使用独立实现 + 包装器
                standalone = StandaloneLLM(
                    base_url=self.ensure_http_prefix(f"{baseurl}/v1"),
                    api_key="lm-studio",
                    model=model,
                    temperature=temperature
                )
                return LangChainWrapper(standalone)
                
            elif backend == 'openai':
                if not baseurl:
                    raise ValueError("OpenAI需要API密钥")
                
                # 使用独立实现 + 包装器
                standalone = StandaloneLLM(
                    base_url= "https://api.openai.com/v1",
                    api_key=baseurl,
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
################## initial ##############
    def __init__(self,local):
        self.torch = torch
        self.conversation_html = ""
        # 初始化配置
        self._init_config()
        self._init_variables()
        self._init_device()
        self.client = WorkingLLMClient()
        if not local:
            import webui as wui
            #webUI
            self.myui=wui.ChatUI(self)
            #self.myui.setup_routes()
            self.myui.runui()

        else:
            import locui as lui
            # 创建GUI
            #localUI
            self.myui = lui.LocalUI()
            self.myui.runui(self)
            # 加载已有数据
            self.load_existing_data()
            self.myui.root.mainloop()

    def on_close(self):
        self.save_vectorstore()
        self.save_settings()
        self.myui.destroyui()

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
        self.pdf_paths = [] # 所有pdf文件完全路径
        self.failed_files = {} # 所有操作失败的pdf文档列表
        self.file_hashes = {} # 以pdf路径做索引的文件hash列表，存在 {hash_db_path}
        self.vectorstore = None # 合成的向量总库，存在 {vectorstore_path}\index
        self.qa_chain = None # QA链
        self.individual_vectorstores = {} # 以pdf路径做索引的单个文件向量列表，存在 {vectorstore_path}\{hash}
        self.conversation_history = [] # 对话历史
        self.max_token=1000000

        # 模型相关
        self.ollama_models = ["llama2"]
        self.default_prompt = """You are an expert AI assistant specialized in providing accurate, context-based responses. Your role is to analyze the provided context and answer questions with precision and clarity.
        1. Start with a direct answer if possible
        2. Support with evidence from the context
        3. Note any limitations or assumptions
        4. Suggest follow-up questions if relevant
        5. When there are Priority Documents selected,  Do not mix the content from other documents """   

        # 消息系统
        self.message_queue = queue.Queue()

    def _init_device(self):
        """初始化设备"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"运行设备: {self.device.upper()},开始初始化embedding...")
        self.embeddings = self.creat_embedding_model()
        self.message_queue.put(("message", "系统", f"检测到运行设备: {self.device.upper()}"))
    
    def load_existing_data(self):
        self.myui.disable_send_btn()
        print("加载已有数据...")
        """加载已有数据"""
        # 加载设置
        self.load_settings()
        # 尝试加载向量库
        print("寻找已有向量数据库" + self.vectorstore_path)
        if os.path.exists(self.vectorstore_path):
            print("正在加载已有向量数据库...")
            self.message_queue.put(("message", "系统", "正在加载已有向量数据库..."))
            if self.load_vectorstore():
                print("向量数据库加载成功...")
                self.message_queue.put(("message", "系统", "向量数据库加载成功"))
                self.load_pdf_list()
                self.myui.update_system_ready_message()
            else:
                print("向量数据库加载失败...")
                self.message_queue.put(("message", "系统", "向量数据库加载失败"))
        else:
            print("未找到已有数据...")
            self.message_queue.put(("status", "准备就绪，请导入PDF文件"))

    def load_pdf_list(self):
        """加载已有PDF列表到管理界面,顺便严重获取的文件路径对于的文件是否存在，不存在时删hash库，总向量库，和子向量库"""
        print("加载已有PDF列表到管理界面")
        self.myui.clear_pdf_list()
        self.pdf_paths = []
        
        # 从文件哈希记录中加载PDF列表
        valid_pdfs = []
        for pdf_path in self.file_hashes.keys():# 在加载向量库时载入的
            if os.path.exists(pdf_path):
                valid_pdfs.append(pdf_path) # 保留有效文件
            else:
                print(f"文件不存在，已移除: {pdf_path}") # 更新库内容，去除无效文件
                del self.file_hashes[pdf_path] # 是否需要存一下？
                try:
                    del self.individual_vectorstores[pdf_path]
                    self.delete_vectors_by_metadata_path(pdf_path)
                except Exception as e:
                    print (f"很奇怪，文件在硬盘hash库里，却没有单独向量库或不在总向量库里{str(e)}")
                
        # 检查哪些PDF已有向量库
        loaded_count = 0
        for pdf_path in valid_pdfs:# 把有效文件加入文件列表以及其关联的数组
            filename = os.path.basename(pdf_path)
            file_info = {
                    'name': filename,
                    'size': os.path.getsize(pdf_path),
                    'path': pdf_path
                }
            try:
                # 检查是否已有向量库
                if pdf_path in self.individual_vectorstores:
                    loaded_count += 1
                    self.pdf_paths.append(pdf_path)
                    self.myui.insert_pdf_list(file_info)
                    
                else:
                    print (f"很奇怪，文件在硬盘hash库里，却没有单独向量库,把这个文件列到一个数组里重新向量化一下")
                    self._process_added_pdfs(pdf_path)
            except Exception as e:
                print(f"万一数据库数组里有这个文件了，也就别往列表里装了，不然重复了{str(e)}")
        
        if valid_pdfs:
            print(f"已加载 {loaded_count}/{len(valid_pdfs)} 向量库/PDF文件到列表")
        else:
            print("没有可用的PDF文件")

    def creat_embedding_model(self):#TODO 看看能否搞成Vulkan优化
        """创建嵌入模型"""
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': self.device},
            encode_kwargs={
                'batch_size': 32,
                'device': self.device,
                'normalize_embeddings': True
            }
        )

#################### process PDF files #########################

    def process_pdfs(self, folder):
        '''递归扫描所有子文件夹内的PDF文件'''
        try: 
            added_count = 0
            skipped_count = 0
            failed_count = 0
            for root, _, files in os.walk(folder):
                result = self._process_added_pdfs(files, root)
                added_count += result[0]
                skipped_count += result[1]
                failed_count += result[2]

            # 合并向量库
            if added_count > 0:
                self.vectorstore = self.merge_vectorstores(self.individual_vectorstores,self.vectorstore)
                if self.vectorstore:
                    # 保存合并后的向量库
                    self.save_vectorstore()
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
                self.myui.update_system_ready_message()
        except Exception as e:
            error_msg = f"处理PDF时出错: {str(e)}"
            self.message_queue.put(("error", error_msg))
        finally:
            self.message_queue.put(("enable_buttons", True))
 
    def _process_added_pdfs(self, files, folder=None ):
        """实际处理新增多个PDF文件的逻辑
        
        Args:
            files (list): PDF文件路径listh或者PDF文件名list
            folder (str|None): None或者PDF共同的文件夹名
            
        Returns:
            array: 处理结果，[成功，跳过，失败]
        """
        try:
            added_count = 0
            skipped_count = 0
            failed_count = 0
            
            for file_path in files:
                file_path = os.path.join(folder, file_path) if folder else file_path
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
                    file_info = {
                            'name': os.path.basename(file_path),
                            'size': os.path.getsize(file_path),
                            'path': file_path
                        }
                    # 更新列表显示
                    self.message_queue.put(("update_list", file_info))
                    added_count += 1
                else:
                    failed_count += 1

            if folder:
                '''被递归文件夹调用,则返回'''
                return [added_count,skipped_count,failed_count]
 
            # 合并向量库
            if added_count > 0:
                self.vectorstore = self.merge_vectorstores(self.individual_vectorstores,self.vectorstore)
                if self.vectorstore:
                    # 保存合并后的向量库
                    self.save_vectorstore()

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
                self.myui.update_system_ready_message()
        
        except Exception as e:
            error_msg = f"添加PDF文件时出错: {str(e)}"
            print(error_msg)
            self.message_queue.put(("error", error_msg))
            self.message_queue.put(("enable_button", "add_btn"))

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
                
            # 读取PDF内容转成list并且添加metadata
            reader = PdfReader(pdf_path)
            documents = []
            for  page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": pdf_path, "page": page_num} #组成：路径，页码
                    ))
                    # print (page_num)
            
            if not documents:
                self.failed_files[pdf_path] = "无法提取文本内容"
                print(f"无法提取文本内容: {pdf_path}")
                return False
            
            # 分割PDF的list
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
            
            # 为当前PDF创建单独的向量库
            # vectorstore = FAISS.from_texts(chunks, embeddings)
            vectorstore = FAISS.from_documents(
                split_docs,  # 使用Document对象而非纯文本
                self.embeddings
            )
            # 保存到单独向量库字典
            self.individual_vectorstores[pdf_path] = vectorstore
            
            # 更新文件哈希记录
            self.file_hashes[pdf_path] = file_hash
            
            print(f"成功处理PDF文件: {pdf_path} (生成 {len(documents)} 个文本块)")
            print(f"文档条数：:{len(vectorstore.index_to_docstore_id)}")
            print(f"向量维度：{vectorstore.index.d}")
            print(f"向量数量：{vectorstore.index.ntotal}")
            return True
            
        except PdfReadError as e:
            self.failed_files[pdf_path] = "PDF文件损坏或加密"
            print(f"PDF读取错误: {pdf_path} - {str(e)}")
            return False
        except Exception as e:
            self.failed_files[pdf_path] = str(e)
            print(f"处理PDF文件出错: {pdf_path} - {str(e)}")
            return False

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
                    if file_path in self.file_hashes:
                        self.remove_saved_individual_vec(file_path)
                        del self.file_hashes[file_path]
                    if file_path in self.individual_vectorstores:
                        del self.individual_vectorstores[file_path]
                    self.delete_vectors_by_metadata_path(file_path)
                    # 从列表框中移除
                    self.message_queue.put(("remove_from_list", i))
                    del self.pdf_paths[i]

                # 保存更新后的数据
                self.save_vectorstore()
                # 准备结果消息
                result_msg = (
                    f"已成功移除 {len(removed_files)} 个PDF文件:\n"
                    + "\n".join(f"• {name}" for name in removed_files)
                )
                
                # 更新系统状态
                self.message_queue.put(("message", "系统", result_msg))
                self.myui.update_system_ready_message()
            
        
        except Exception as e:
            error_msg = f"移除PDF文件时出错: {str(e)}"
            self.message_queue.put(("error", error_msg))
        
        finally:
            # 恢复按钮状态
            self.message_queue.put(("enable_button", "remove_btn"))
            self.message_queue.put(("update_status", "准备就绪"))

#################################### GUI service #########################
    def queue_Empty(self):
        return queue.Empty

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
        selected_indices = self.myui.selected_pdf_list_count()
        
        if not selected_indices:
            self.myui.update_status("")
            return
        
        # 只处理第一个选中的文件
        selected_index = selected_indices[0]
        
        if selected_index < len(self.pdf_paths):
            file_path = self.pdf_paths[selected_index]
            
            # 调用独立的提取函数获取预览内容
            preview_text = self.get_pdf_preview(file_path,100)
            
            # 显示文件信息
            file_name = os.path.basename(file_path)
            info_text = f"选中的文件：{file_name}{self.myui.nline}路径：{file_path}{self.myui.nline}内容预览：{preview_text}"
            self.myui.update_status(info_text)

##################################### conversation service #######################
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
            summary = self.myui.llm.generate(prompts=[summary_prompt]).generations[0][0].text
            print("历史记录3~10被总结")
            return f"""较早对话总结：{summary}
                        最近对话记录：
                        {chr(10).join(recent_actual)}"""
            
        except Exception as e:
            print(f"生成总结时出错：{e}")
            return "最近对话：\n" + "\n".join(recent_actual) 

    def process_source(self,result):
        # 处理来源信息
        print("show sources")
        sources = []
        for doc in result:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source = f"[{os.path.basename(doc.metadata['source'])}]({doc.metadata['source']})\n"
                if source not in sources:
                    sources.append(source)
        # 构建响应
        if sources:
            return f"\n\n来源: {', '.join(sources)}"
        return ""

    def get_answer(self, question):
        try:
            
            enhanced_question = ""
            answer_source = ""
            if not self.myui.get_conv_var():
                if self.qa_chain is None:
                    return
                #RAG search
                target_file = ""
                # 构建完整的查询内容
                query_parts = []
                # 构建增强问题指向相应文件
                print("try target some pdf")
                selected_indices = self.myui.selected_pdf_list_count()
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
                self.myui.update_conversation("User", question)
                # 调用 QA 链
                result = self.qa_chain.invoke({"query": enhanced_query})
                query_answer = result["result"]
                answer_source =  self.process_source(result["source_documents"])
                if not self.myui.get_check_var():
                    self.myui.update_conversation("AI",f"回答:{query_answer} {answer_source}")
                    self.message_queue.put(("enable_buttons", True))
                    return
                
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
                else:
                    self.myui.update_conversation("AI",f"回答:{query_answer} {answer_source}")
                    self.message_queue.put(("enable_buttons", True))
                    return
                # 组合完整提问
                enhanced_question = "\n\n".join(question_parts) 
            else:
                #仅仅对话
                enhanced_question = question
            try:
                system_prompt = self.myui.get_system_prompt_var()
                #messages = [
                #           {"role": "system", "content": system_prompt},
                #            {"role": "user", "content": enhanced_question}
                #       ]
                messages =f"{system_prompt}\n\n User:{enhanced_question} "

                answer = self.myui.llm.generate(prompts=[messages]).generations[0][0].text
                if not self.myui.get_check_var():
                    # RAG search
                    self.myui.update_conversation("AI", f"回答: {answer} {answer_source}")
                
            except Exception as e:
                print(f"生成总结时出错：{e}")
               
        except Exception as e:
            print(f"获取答案时出错: {str(e)}")
            self.message_queue.put(("error", f"获取答案时出错: {str(e)}"))
        finally:
            self.message_queue.put(("enable_buttons", True))

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

############################## LLM service #########################

    def apply_prompt(self):
        print("重置QA chain")
        self.qa_chain = None

############################ hash service #########################
    def calculate_file_hash(self, filepath):
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    def calculate_hash(self,content):
        pass


    def save_hashes(self):
        if hasattr(self,'file_hashes') and self.file_hashes:
            with open(self.hash_db_path, 'w') as f:
                json.dump(self.file_hashes, f)

################################ vector store ######################
    def save_vectorstore(self):
        path = self.vectorstore_path
        print(f"保存向量数据库到 {path}...")
        try:
            os.makedirs(path, exist_ok=True)
            
            # 保存合成向量库
            if self.vectorstore:
                self.vectorstore.save_local(
                    folder_path=path,
                    index_name="index"
                )
                print("保存总向量数据库成功！")
                print(f"文档条数：{len(self.vectorstore.index_to_docstore_id)}")
                print(f"向量维度：{self.vectorstore.index.d}")
                print(f"向量数量：{self.vectorstore.index.ntotal}")
            
            # 保存单个向量库
            if hasattr(self, 'individual_vectorstores') and self.individual_vectorstores:
                # 假设self.individual_vectorstores是字典 {file: vectorstore}
                for file, vectorstore in self.individual_vectorstores.items():
                    if file in self.file_hashes:
                        name = self.file_hashes[file]
                        vectorstore.save_local(
                            folder_path=path,
                            index_name=name
                        )
                self.save_hashes()
            return True
            
        except Exception as e:
            print(f"保存向量数据库失败: {str(e)}")
            return False

    def load_vectorstore(self):
        path = self.vectorstore_path
        print(f"从 {path} 加载向量数据库...")
        try:
            if not os.path.exists(path):
                print("向量数据库目录不存在")
                return False
            
            # 合成向量库加载
            main_index_path = os.path.join(path, "index.faiss")
            if os.path.exists(main_index_path):
                self.vectorstore = FAISS.load_local(
                    folder_path=path,
                    index_name="index",
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"文档条数：{len(self.vectorstore.index_to_docstore_id)}")
                print(f"向量维度：{self.vectorstore.index.d}")
                print(f"向量数量：{self.vectorstore.index.ntotal}")
            else:
                print("合成向量库不存在")
            
            # 加载单文件向量
            if os.path.exists(self.hash_db_path):
                with open(self.hash_db_path, 'r', encoding='utf-8') as f:
                    self.file_hashes = json.load(f)
                print(f"{len(self.file_hashes)}条单文件向量")
                self.individual_vectorstores = {}
                if self.file_hashes:
                    for file, hash_val in self.file_hashes.items():
                        index_path = os.path.join(path, f"{hash_val}.faiss")
                        # print(f"导入: {file} 对应的向量文件 {hash_val}.faiss 中")
                        if os.path.exists(index_path):
                            self.individual_vectorstores[file] = FAISS.load_local(
                                folder_path=path,
                                index_name=hash_val,
                                embeddings=self.embeddings,
                                allow_dangerous_deserialization=True
                            )
                        else:
                            print(f"警告: {file} 对应的向量文件 {hash_val}.faiss 不存在")
                else:
                    print("文件-hash存档为空，所以没载入单个文件的向量库")
            return True
            
        except Exception as e:
            print(f"加载向量数据库失败: {str(e)}")
            return False

    def merge_vectorstores(self, individual_vectorstores,vectorstores = None):
        """合并所有单独的向量库到主向量库"""
        if not individual_vectorstores:
            self.message_queue.put(("message", "系统", "没有可用的向量库"))
            return None
        try:
            print("开始合并向量库...")
            start_time = datetime.now()
            first_path = ""
            if not vectorstores:
                # 如果没有已存在的向量库，获取第一个向量库作为基础
                first_path = next(iter(individual_vectorstores))
                vectorstores = individual_vectorstores[first_path]

            # 合并剩余的向量库
            merge_count = 0
            if vectorstores:
                for path, vectorstore in individual_vectorstores.items():
                    if (path != first_path) :
                        try:
                            vectorstores.merge_from(vectorstore)
                            merge_count += 1
                        except Exception as e:
                            print(str(e))
            
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"向量库合并完成，共合并 {merge_count + 1} 个向量库，耗时 {elapsed:.2f} 秒")
            
            # 更新系统信息
            self.message_queue.put(("message", "系统", 
                                  f"向量库合并完成，共包含 {len(individual_vectorstores)} 个PDF文档"))

            return vectorstores
            
        except Exception as e:
            error_msg = f"合并向量库时出错: {str(e)}"
            print(error_msg)
            self.message_queue.put(("error", error_msg))
            return None

    def delete_vectors_by_metadata_path(self, target_path):
        """
        删除 metadata 中 path 为指定值的所有向量
        :param db: FAISS 向量库对象
        :param target_path: 要匹配的 source 值（如 "/data/file1.PDF")
        :return: 被删除的文档数量
        """
        # 收集需要删除的文档 ID
        ids_to_delete = []
        if not self.vectorstore:
            return
        for doc_id, doc_meta in self.vectorstore.docstore.__dict__.items():
            for id, doc in doc_meta.items():
                path = doc.metadata["source"]
                if path == target_path:
                    ids_to_delete.append(id)
        
        # 执行批量删除
        if ids_to_delete:
            self.vectorstore.delete(ids_to_delete)
            print(f"已删除 {len(ids_to_delete)} 个文档")
        else:
            print("未找到匹配的文档")
        
        return len(ids_to_delete)

    def remove_saved_individual_vec(self,target):
        hash = self.file_hashes[target]
        faiss = os.path.join(self.vectorstore_path,f"{hash}.faiss")
        pkl = os.path.join(self.vectorstore_path,f"{hash}.pkl")
        if os.path.exists(faiss) and os.path.exists(pkl):
            os.remove(faiss)
            os.remove(pkl)

############################### settings ##########################
    def save_settings(self):
        """保存当前设置"""
        settings = {
            "backend": self.myui.get_backend_var(),
            "model": self.myui.get_model_var(),
            "conf": self.myui.get_config_var(),
            "temperature": self.myui.get_temp_var(),
            "max_tokens": self.myui.get_max_tokens_var(),
            "last_pdf_folder": self.myui.get_pdf_folder_var(),
            "prompt": self.myui.get_system_prompt_var()
        }
        
        try:
            with open(self.settings_path, 'w') as f:
                json.dump(settings, f)
            self.message_queue.put(("系统", "设置已保存"))
        except Exception as e:
            self.message_queue.put(("系统", f"保存设置失败: {str(e)}"))

    def load_settings(self):
        """加载保存的设置"""
        print("加载设置...")

        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, 'r') as f:
                    settings = json.load(f)
                
                self.myui.set_model_var(settings.get("model", "llama2"))
                self.myui.set_backend_var(settings.get("backend",'ollama'))
                self.myui.set_config_var(settings.get("conf","http://localhost:11434"))
                self.myui.set_temp_var(settings.get("temperature", 0.7))
                self.myui.set_max_tokens_var(settings.get("max_tokens", 2000))
                self.myui.set_pdf_folder_var(settings.get("last_pdf_folder", ""))
                self.myui.set_system_prompt_var(settings.get("prompt", self.default_prompt))
                print("设置加载成功！")
                return True
        except Exception as e:
            print(f"加载设置失败: {str(e)}")
        return False

    def parent_print(self,var):
        print(var)


def main(local_mode=True):
    app = PDFRAGApp(local_mode)
    # ... 其他逻辑

if __name__ in {'__main__', '__mp_main__'}:
    parser = argparse.ArgumentParser(description='PDF RAG 系统参数配置')
    parser.add_argument('-l', '--local', action='store_true', help='启动本地界面')
    args = parser.parse_args()
    main(args.local)  # 调用主函数