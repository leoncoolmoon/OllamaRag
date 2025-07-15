from tkinter import ttk, filedialog, scrolledtext, messagebox
import tkinter as tk
import os #用于打开pdf打开文件夹之类的界面本地操作
from threading import Thread
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from datetime import datetime
import webbrowser
import re
import sys

class LocalUI:
#################### init #############################
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("高级PDF RAG系统 v9 (Ollama,lm studio)")
        self.root.geometry("1300x850")
        # 启动消息处理
        self.root.after(100, self.process_messages)
        self.last_selection = [] # 上回选取的pdf
        self.nline="\n"
    def runui(self,parent):
        self.parent = parent
        # UI 相关
        self.model_var = tk.StringVar(value=self.parent.ollama_models[0] if self.parent.ollama_models else "llama2")
        self.temp_var = tk.DoubleVar(value=0.7)
        self.max_tokens_var = tk.IntVar(value=2000)
        self.check_var = tk.BooleanVar() #是否包含历史对话
        self.conv_var = tk.BooleanVar() #是否仅仅对话不进行搜索
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.parent.on_close)

    def destroyui(self):
        self.root.destroy()
        sys.exit(0)
##################### gui ######################################
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

        button_panel = ttk.Frame(left_panel)
        button_panel.pack(fill=tk.X, pady=(5, 0))
        
        self.hx_btn = ttk.Checkbutton(
            button_panel,
            text="包含之前对话内容",
            width=25,
            variable=self.check_var,
            onvalue=True,
            offvalue=False
        )
        self.hx_btn.pack(side=tk.LEFT, padx=2)
        self.conv = ttk.Checkbutton(
            button_panel,
            text="仅仅对话",
            width=25,
            variable=self.conv_var,
            onvalue=True,
            offvalue=False
        )
        self.conv.pack(side=tk.LEFT, padx=2)
        clear_btn = ttk.Button(
            button_panel,
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

    def _create_pdf_notebook(self, mparent):
        """创建PDF管理笔记本"""
        self.pdf_notebook = ttk.Notebook(mparent)
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
        self.pdf_list.bind("<<ListboxSelect>>", self.parent.show_selected_pdf_info)

        
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

    def _create_model_settings(self, mparent):
        """创建带标签页的模型和提示设置区域"""
        # 创建Notebook（标签页容器）
        notebook = ttk.Notebook(mparent)
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
            values=["ollama", "lmstudio"],# "openai"],
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
            to=self.parent.max_token,
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
        self.prompt_input.insert(tk.END, self.parent.default_prompt)
        # 应用提示词
        self.apply_prompt_btn = ttk.Button(
            prompt_tab,
            text="应用",
            command=self.parent.apply_prompt,
            width=8
        )
        self.apply_prompt_btn.pack(side=tk.RIGHT, padx=2)

    def _create_system_info(self, mparent):
        """创建系统信息区域"""
        info_frame = ttk.LabelFrame(mparent, text="系统信息", padding="10")
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
            command=self.parent.save_settings,
            width=15
        )
        save_btn.pack(pady=5)

########################### gui service ####################
    def disable_send_btn(self):
        self.send_btn.config(state=tk.DISABLED)
    def clear_pdf_list(self):
        self.pdf_list.delete(0, tk.END)
    def insert_pdf_list(self,file_info):
        self.pdf_list.insert(tk.END,file_info['name'])
    def process_messages(self):
        try:
            while True:
                try:
                    msg_type, *content = self.parent.message_queue.get_nowait()
                    
                    if msg_type == "status":
                        self.update_status(content[0])
                    elif msg_type == "message":
                        self.parent.message_queue.put(content[0], content[1])
                    elif msg_type == "error":
                        self.parent.message_queue.put(("错误", content[0]))
                        self.status_var.set(content[0])
                    elif msg_type == "enable_buttons":
                        self.import_btn.config(state=tk.NORMAL)
                        self.browse_btn.config(state=tk.NORMAL)
                        self.send_btn.config(state=tk.NORMAL)
                    #add pdf
                    elif msg_type == "update_list":
                        self.insert_pdf_list(content[0])
                    elif msg_type == "enable_button":
                        if content[0] == "add_btn":
                            self.add_btn.config(state=tk.NORMAL)
                        elif content[0] == "remove_btn":
                            self.remove_btn.config(state=tk.NORMAL)
                        elif content[0] == "send_btn":
                            self.send_btn.config(state=tk.NORMAL)
                    #delete pdf
                    elif msg_type == "remove_from_list":
                        self.pdf_list.delete(content[0])
                except self.parent.queue_Empty():
                    break
        finally:
            self.root.after(100, self.process_messages)

    def update_status(self, message):
        self.status_var.set(message)
    def selected_pdf_list_count(self):
        return self.pdf_list.curselection()

    def select_all_pdfs(self):
        """选择列表框中的所有PDF文件"""
        #self.pdf_list.selection_clear(0, tk.END)  # 先清除现有选择
        self.pdf_list.selection_set(0, tk.END)    # 选择所有项目
        
        
        # 更新状态显示（可选）
        selected_count = len(self.pdf_list.curselection())
        self.update_status(f"已选择 {selected_count} 个PDF文件")
        # 可选：滚动到列表底部确保所有选项可见
        if selected_count > 0: #self.pdf_list.size()
            self.pdf_list.see(tk.END)

    def open_selected_pdf(self, event):
        """双击打开选中的PDF文件"""
        # 获取选中的索引
        selected_indices = self.pdf_list.curselection()
        
        if not selected_indices:
            return
        
        # 只处理第一个选中的文件（双击通常只选中一个）
        selected_index = selected_indices[0]
        
        # 检查索引是否有效
        if selected_index < len(self.parent.pdf_paths):
            file_path = self.parent.pdf_paths[selected_index]
            
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
                self.parent.message_queue.put(("错误", f"无法打开PDF文件: {str(e)}"))

    def select_non_pdfs(self):
        """取消选择列表框中的PDF文件"""
        self.pdf_list.selection_clear(0, tk.END)  # 先清除现有选择
        # 更新状态显示（可选）
        selected_count = 0
        self.update_status(f"已取消选择 PDF文件")

    def clear_conversation(self):
        """清空对话历史"""
        self.conversation.delete(1.0, tk.END)
        self.parent.conversation_history = []
        self.parent.message_queue.put(("系统", "对话历史已清空"))

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
            self.parent.message_queue.put(("系统", "请先选择要删除的PDF文件"))
            return
        
        # # 确认对话框
        # if not messagebox.askyesno(
        #     "确认删除",
        #     f"确定要删除选中的 {len(selected_indices)} 个PDF文件吗？\n"
        #     "此操作将同时移除相关的向量数据且不可恢复！"
        # ):
        #     return
        
        # 禁用按钮防止重复操作
        self.remove_btn.config(state=tk.DISABLED)
        self.update_status(f"正在移除 {len(selected_indices)} 个PDF文件...")
        
        # 在新线程中执行移除操作
        Thread(target=self.parent._process_removal, args=(selected_indices,), daemon=True).start()

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
        Thread(target=self.parent._process_added_pdfs, args=(files,), daemon=True).start()

    def import_pdfs(self):
        folder = self.pdf_folder_var.get()
        if not folder:
            self.update_status("请先选择PDF文件夹")
            return
        
        # 清空列表
        self.parent.pdf_paths = []
        self.pdf_list.delete(0, tk.END)
        self.parent.failed_files = {}
        
        # 禁用按钮防止重复点击
        self.import_btn.config(state=tk.DISABLED)
        self.browse_btn.config(state=tk.DISABLED)
        self.update_status("正在扫描PDF文件...")
        
        # 在新线程中处理PDF
        Thread(target=self.parent.process_pdfs, args=(folder,), daemon=True).start()

    def update_system_ready_message(self):
        """更新系统就绪消息"""
        model = self.model_var.get()
        pdf_count = len(self.parent.pdf_paths)
        self.model_info_var.set(f"当前模型: {model}")
        self.vectorstore_info.set(f"向量库: 已加载 ({len(self.parent.individual_vectorstores)}个)")
        self.pdf_count_info.set(f"已加载PDF: {pdf_count}个")
        self.last_update_info.set(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        message = (
            f"RAG系统已准备好回答关于PDF内容的问题\n\n"
            f"当前模型: **{model}**\n"
            f"已加载PDF文档: **{pdf_count}**个\n"
            f"最后更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.parent.message_queue.put(("系统", message))
        self.send_btn.config(state=tk.NORMAL)  # 添加这行

    def force_reprocess(self):
        """强制重新处理所有文件"""
        # 重新从文件夹开始生成向量库
        self.parent.file_hashes = {}
        self.pdf_list.delete(0, tk.END)
        self.parent.pdf_paths = []
        self.parent.individual_vectorstores = {}
        self.parent.vectorstore = None
        
        folder = self.pdf_folder_var.get()
        if folder:
            self.import_pdfs()
        else:
            self.parent.message_queue.put(("系统", "请先选择PDF文件夹"))

    def send_message(self, event=None):
        #没有向量库
        if not hasattr(self.parent, 'vectorstore') or self.parent.vectorstore is None:
            self.parent.message_queue.put(("message", "请先成功导入PDF文件或加载向量库"))
            return

        user_text = self.user_input.get("1.0", tk.END).strip()
        #没有用户提示就不发送
        if not user_text:
            return
        #没有选定模型也不发送
        print(self.model_var.get())
        print(self.config_var.get())
        if not self.model_var.get().strip():
            self.parent.message_queue.put(("message", "请先选定模型"))
            return
        #if not hasattr(self, 'qa_chain') or self.qa_chain is None:
        if not self.get_check_var():
            try:
                # 获取用户设置的系统提示
                system_prompt = self.prompt_input.get("1.0", tk.END)
                if not system_prompt:
                    system_prompt = "You are an helpful expert AI assistant"
                    # 如果有自定义系统提示，使用增强版本
                prompt_template = f"""
                                    {system_prompt}

                                    You are answering based on the provided context. Please follow these additional guidelines:
                                    - Base your answers strictly on the provided context, Focus on the AUTHOR'S OWN TEXT, not citations or the paper's references
                                    - Reference previous conversation when relevant, but prioritize current context  
                                    - Quote specific evidence and explain your reasoning, Check the metadate and add page number to the evidence you find from RAG search result 
                                    - Express confidence levels based on available information
                                    - If information is missing, state this clearly

                                    Context: {{context}}

                                    Question: {{question}}

                                    Answer:"""

                PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )                
                
                # 指向性QA链
                
                selected_indices = self.pdf_list.curselection()
                target_vectors_merge = None
                self.llm=self.parent.client.llmClient(
                    backend=self.get_backend_var(),
                    baseurl=self.get_config_var(),
                    model=self.get_model_var()#,
                    #temperature=self.localui.get_temp_var()
                )
                self.parent.parent_print(f"{self.get_backend_var()},\n {self.get_config_var()},\n {self.get_model_var()} \n\n {self.llm}")
                if selected_indices and self.llm :# 如果选中
                    if self.last_selection != selected_indices:# 如果选中有变化
                        self.last_selection = selected_indices # 上回选取的pdf
                        target_vectors = {key: self.parent.individual_vectorstores[key] for key in (self.parent.pdf_paths[i]  for i in selected_indices)}
                        target_vectors_merge = self.parent.merge_vectorstores(target_vectors) 
                        if target_vectors_merge:# 向量合并成功
                            self.parent.qa_chain = RetrievalQA.from_chain_type(
                                self.llm,
                                chain_type="stuff",
                                retriever=target_vectors_merge.as_retriever(search_kwargs={"k": 5}), #其实这样可能会导致文档间的不均衡
                                return_source_documents=True,
                                chain_type_kwargs={"prompt": PROMPT},
                                verbose=True
                            )
                        else:# 向量没有合并成功
                            self.parent.qa_chain = RetrievalQA.from_chain_type(
                                self.llm,
                                chain_type="stuff",
                                retriever=self.parent.vectorstore.as_retriever(search_kwargs={"k": 5}), #其实这样可能会导致文档间的不均衡
                                return_source_documents=True,
                                chain_type_kwargs={"prompt": PROMPT},
                                verbose=True
                            )
                else:# 没有选中
                    self.parent.qa_chain = RetrievalQA.from_chain_type(
                        self.llm,
                        chain_type="stuff",
                        retriever=self.parent.vectorstore.as_retriever(search_kwargs={"k": 5}), #其实这样可能会导致文档间的不均衡
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": PROMPT},
                        verbose=True
                    )
            except Exception as e:
                print(f"设置QA链时出错: {str(e)}")
                raise
        else:
            pass
        self.user_input.delete("1.0", tk.END)
        self.send_btn.config(state=tk.DISABLED)
        Thread(target=self.parent.get_answer, args=(user_text,), daemon=True).start()

    def get_config_var(self):
        return self.config_var.get()
    def set_config_var(self,var):
        self.config_var.set(var)

    def get_backend_var(self):
        return self.backend_var.get()
    def set_backend_var(self,var):
        self.backend_var.set(var)
    
    def get_model_var(self):
        return self.model_var.get()
    def set_model_var(self,var):
        self.model_var.set(var)
    
    def get_temp_var(self):
        return self.temp_var.get()
    def set_temp_var(self,var):
        rounded = round(float(var), 2)
        self.parent.parent_print(f"set temp as {var}, after cal {rounded}")
        self.temp_var.set(rounded)
        self.temp_slider.set(rounded)  # ✅ 手动同步滑块
    
    def update_temp_value(self, val):
        self.temp_var.set(round(float(val), 2)) 
    
    def update_temp_slider(self, event=None):
        try:
            temp = float(self.temp_var.get())
            if temp < 0 or temp > 1:
                raise ValueError
            self.set_temp_var(temp)
        except ValueError:
            messagebox.showerror("错误", "请输入0到1之间的数字")
            self.set_temp_var(0.7)

    def browse_folder(self):
        folder = filedialog.askdirectory(title="选择包含PDF的文件夹")
        if folder:
            self.pdf_folder_var.set(folder)
    def get_check_var(self):
        return self.check_var.get()
    def get_conv_var(self):
        return self.conv_var.get()

    def get_system_prompt_var(self):
        return self.prompt_input.get("1.0", tk.END)
    def set_system_prompt_var(self,var):
        self.prompt_input.setvar(var)
    
    def get_max_tokens_var(self):
        return self.max_tokens_var.get()
    def set_max_tokens_var(self,var):
        self.max_tokens_var.set(var)
    
    def get_pdf_folder_var(self):
        return self.pdf_folder_var.get()
    def set_pdf_folder_var(self,var):
        self.pdf_folder_var.set(var)

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
        self.parent.conversation_history.append((sender, re.sub(r'<think>.*?</think>', '', message, flags=re.DOTALL)))
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
    def update_model_list(self, event=None):
        mbackend = self.backend_dropdown.get() 
        self.ollama_models = self.parent.client.get_available_models(mbackend)
        self.parent.message_queue.put(("模型","self.ollama_models"))
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

if __name__ == "__main__":
    print("⚠️ 请不要直接运行此模块，正在跳转到主程序...")
    import sys
    from RAGv9 import main  # 导入主程序的 main 函数
    
    # 手动模拟 argparse
    if "--local" in sys.argv or "-l" in sys.argv:
        main(local_mode=True)  # 直接调用主函数
    else:
        main(local_mode=False)