from nicegui import ui, app
from nicegui.elements.scroll_area import ScrollArea
from nicegui.events import UploadEventArguments
from typing import Optional
import os #用于打开pdf打开文件夹之类的界面本地操作
from threading import Thread
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from datetime import datetime
import re
import sys

class ChatUI:
########## init ##########
    def __init__(self,parent):
        self.parent=parent
        self.upload_path = "uploads"
        self.check_var = False
        self.conv_var = False
        self.default_prompt = "默认系统提示词"
        self.max_tokens_var = 2000
        self.backend_var = "ollama"
        self.model_var = ""
        self.temp_var = 0.7
        self.config_var = ""
        self.status_var = "准备就绪"
        self.model_info_var = ""
        self.vectorstore_info = "向量库: 未加载"
        self.pdf_count_info = "已加载PDF: 0"
        self.last_update_info = "最后更新: 无"
        self.pdf_folder_var = ""
        self.uploaded_files = []  # 全局已上传文件记录 ##TODO 有逻辑问题，这里的uploaded_files应该只用来记录上传文件情况，现在和主程序里的self.pdf_paths 混了，self.pdf_paths 应该是同步了列表的
        self.selecte_items = {}  # key: 文件名, value: checkbox对象
        self.last_selection = [] # 上回选取的pdf
        self.nline="<br>"

    def define_path(self,path):
        if not os.path.exists(path): 
            os.makedirs(path, exist_ok=True)

    def setup_routes(self):
        ui.timer(0.1, self.process_messages, once=True)
        
        # 确保上传目录存在 self.pdf_folder_var
        self.define_path(self.upload_path)

        app.add_static_files('/uploads', self.upload_path)

        ui.add_head_html("""
        <style>
            body { font-size: 14px; }
            html, body { height: 100%; margin: 0; }
        </style>
        """)

    def runui(self):
        self.create_ui()
        self.parent.load_existing_data()
        #self.process_messages()
        self.setup_routes()
        try:
            ui.run(title="高级PDF RAG系统 v9", reload=False, port=8080, show=True)
        except Exception as e:
            print(e)
        finally:
            self.parent.on_close()

    def destroyui(self):
        app.shutdown()
        sys.exit(0)
########## gui ##########
    def create_ui(self):
        """创建整体UI布局"""
        with ui.header().style("background-color: #3874c8; color: white; padding: 10px"):
            with ui.row().classes("w-full justify-between items-center"):
                ui.label().bind_text(self, "status_var")
                ui.label().bind_text(self, "model_info_var")
        with ui.row().classes("w-full h-full"):    
            with ui.column().classes("w-7/12 h-full"):  # 全屏高度的竖排布局
                with ui.row().classes("w-full h-full"):
                    with ui.row().classes("w-full h-[65vh] p-4"):
                        # 对话区域
                        with ui.scroll_area().classes("w-full h-full overflow-auto").props('id="chat_scroll"'):
                            self.conversation = ui.column().classes("w-full space-y-2").props('id="conversation"')
                        with ui.row().classes("w-full items-center"):
                            ui.checkbox("包含之前对话内容").bind_value(self, "check_var")
                            ui.checkbox("仅对话不搜索").bind_value(self, "conv_var")
                            ui.button("清空对话", on_click=self.clear_conversation)
                    # 底部输入区域
                    with ui.row().classes("w-full h-[15vh] "):
                        with ui.row().classes("w-full items-center h-full p-4"):
                            self.user_input = ui.textarea(placeholder="输入消息...").classes("w-[70vw] mr-4").style("height: 50px; resize: none;")
                            self.user_input.on('keydown', self.handle_keydown)
                            self.send_btn=ui.button("发送", on_click=self.send_message).classes("w-24 h-12 justify-right")
            # 右侧控制区域
            with ui.column().classes("w-4/12 h-full p-4"):
                self._create_pdf_notebook()

    def _create_pdf_notebook(self):
        """创建PDF管理标签页"""
        with ui.tabs().classes("w-full") as tabs:
            pdf_manage = ui.tab("PDF文件管理")
            model_tab = ui.tab("模型设置")
            prompt_tab = ui.tab("系统提示词")
            system_info_tab = ui.tab("系统信息")
        
        with ui.tab_panels(tabs, value=pdf_manage).classes("w-full h-full"):
            # PDF文件管理标签页
            with ui.tab_panel(pdf_manage).classes("w-full h-full flex flex-col"):
                # 上传区域
                with ui.column().classes("w-full h-[20%]"):
                    self.add_btn = ui.upload(
                            on_upload=self.handle_upload,
                            on_multi_upload=self.handle_uploads,
                            multiple=True,
                            max_file_size=50_000_000,
                            max_files=10
                        ).props('accept="*"').classes("w-full h-full")
                
                #ui.separator().classes('my-2')
                
                # 按钮区域 - 固定在分隔线下方
                with ui.row().classes("w-full justify-between mb-1"):
                    ui.button("全选", on_click=self.select_all_pdfs).classes("px-4 py-0")
                    ui.button("取消选择", on_click=self.select_non_pdfs).classes("px-4 py-0")
                
                # 文件列表区域
                with ui.column().classes("w-full h-full min-h-0"):
                    ui.label('已上传文件').classes('text-sm mb-1')
                    # 使用滚动容器包裹文件列表
                    #with ui.scroll_area().classes('w-full h-[50vh] flex-grow overflow-y-auto overflow-x-hidden border rounded p-2'):
                    self.pdf_list = ui.column().classes('w-full h-[50vh] flex-grow overflow-y-auto overflow-x-hidden border rounded p-2').props('id="pdf_list"')
            
            # 模型设置标签页
            with ui.tab_panel(model_tab).classes("w-full h-full p-2"):
                with ui.column().classes("w-full h-full space-y-2"):
                    ui.label("后端类型:").classes("text-sm font-semibold mb-0")
                    self.backend_dropdown = ui.select(
                        ["ollama", "lmstudio"], 
                        value="ollama",
                        on_change=lambda e: self.update_model_list()
                    ).bind_value(self, "backend_var").classes("w-full")
                    
                    ui.label("选择模型:").classes("text-sm font-semibold mb-0")
                    self.model_dropdown=ui.select(
                        ["llama2"], 
                        value="llama2",
                        multiple=False
                    ).bind_value(self, "model_var").classes("w-full")
                    
                    ui.label("配置:").classes("text-sm font-semibold mb-0")
                    self.config_var_label = ui.input().bind_value(self, "config_var").classes("w-full")
                    
                    ui.label("温度(0-1):").classes("text-sm font-semibold mb-0")
                    with ui.row().classes("w-full items-center space-x-4"):
                        self.temp_var_input = ui.number(
                            min=0, max=1, step=0.1, 
                            format="%.1f"
                        ).bind_value(self, "temp_var").classes("w-20")
                        self.temp_var_slider = ui.slider(
                            min=0, max=1, step=0.1
                        ).bind_value(self, "temp_var").classes("flex-1")
                    
                    ui.label("最大token数:").classes("text-sm font-semibold mb-0")
                    with ui.row().classes("w-full items-center space-x-4"):
                        self.max_tokens_var_input=ui.number(
                            min=512, max=self.parent.max_token, step=1
                        ).bind_value(self, "max_tokens_var").classes("w-20")
                        self.max_tokens_var_slider=ui.slider(
                            min=512, max=self.parent.max_token, step=1
                        ).bind_value(self, "max_tokens_var").classes("flex-1")
            
            # 系统提示词标签页
            with ui.tab_panel(prompt_tab).classes("w-full h-full p-2"):
                with ui.column().classes("w-full h-full space-y-4 "):
                    # 使用flex布局，让textarea占据剩余空间
                    with ui.column().classes("w-full h-full min-h-0"):
                        self.prompt_input=ui.textarea(
                            value=self.parent.default_prompt,
                            placeholder="请输入系统提示词..."
                        ).classes("w-full h-full").style("resize: none;height: 100% !important; min-height: 300px;")

                    # 按钮区域
                    with ui.row().classes("w-full justify-center mt-2"):
                        ui.button("重置", on_click=self.reset_prompt).classes("mr-2")
                        ui.button("应用", on_click=self.parent.apply_prompt).classes("bg-blue-500 text-white")
            
            # 系统信息标签页
            with ui.tab_panel(system_info_tab).classes("w-full h-full p-2"):
                with ui.column().classes("w-full h-full space-y-4"):
                    ui.label("系统信息").classes("text-sm font-bold mb-1")
                    
                    # 信息展示区域
                    with ui.column().classes("space-y-3 w-full"):
                        with ui.card().classes("w-full p-2"):
                            ui.label("向量库状态").classes("text-sm font-semibold mb-0")
                            ui.label().bind_text(self, "vectorstore_info").classes("text-sm")
                        
                        with ui.card().classes("w-full p-2"):
                            ui.label("文件统计").classes("text-sm font-semibold mb-0")
                            ui.label().bind_text(self, "pdf_count_info").classes("text-sm")
                        
                        with ui.card().classes("w-full p-2"):
                            ui.label("最后更新").classes("text-sm font-semibold mb-0")
                            ui.label().bind_text(self, "last_update_info").classes("text-sm")
                    
                    # 操作按钮区域
                    with ui.row().classes("w-full justify-center mt-6"):
                        ui.button("保存当前设置", on_click=self.parent.save_settings).classes("px-8 py-3 bg-green-500 text-white ")
                        ui.button("重置所有设置", on_click=self.parent.load_settings).classes("px-8 py-3 bg-red-500 text-white ml-4")

########## gui service ##########
    
    def reset_prompt(self):
        """重置提示词到默认值"""
        self.prompt_input.value = self.parent.default_prompt

    def handle_keydown(self,e):
        if e.key == "Enter" and e.ctrl_key:  # 检查是否按下 Ctrl + Enter
            print("Ctrl + Enter pressed!")
            self.send_message()

    async def handle_uploads(self,e):#把指定文件夹一锅端，然后更新列表，如果里面有残留文件可能会有问题
        self.parent.process_pdfs(self.upload_path)
        self.update_file_list()

    async def handle_upload(self,e):#写出上传的文件到指定文件夹
        # In current NiceGUI versions, the file content is available in e.content
        file_content = e.content.read()
        file_name = e.name
        file_type = e.type
        #file_hash =self.parent.calculate_hash(e.content)
        file_path = f"{self.upload_path}/{file_name}"
        with open(file_path, "wb") as f:
            f.write(file_content)

        file_info = {
            'name': file_name,
            'size': len(file_content),
            'type': file_type,
            'path': file_path
        }
        self.uploaded_files.append(file_info)
        # self.parent.process_single_pdf(file_path)#this process will not add file path to pdf_paths

    def make_open_file_callback(self,idx):
        return lambda: ui.download.file(self.parent.pdf_paths[idx])

    def human_readable_size (self,size_in_bytes):
        """将 '1663738 ' 转为 '1.59 MB' 等更友好的单位"""
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        i = 0
        while size_in_bytes >= 1024 and i < len(units) - 1:
            size_in_bytes /= 1024
            i += 1
        return f"{size_in_bytes:.2f} {units[i]}"


    def update_file_list(self):
        self.pdf_list.clear()
        with self.pdf_list:
            for idx,file in enumerate(self.uploaded_files):
                with ui.row().classes('w-full items-center border border-gray-300 rounded') as row:
                    row.on('dblclick', self.make_open_file_callback(idx))
                    
                    self.selecte_items[idx] = ui.checkbox(
                        on_change= self.parent.show_selected_pdf_info
                    ).classes('mr-2') 
                    
                    ui.icon('description').classes('mr-2')
                    
                    # 文件名 label：最多显示两行，超出省略号
                    ui.label(file['name']).classes(
                        'flex-grow text-sm leading-tight overflow-hidden'
                    ).style(
                        '''
                        display: -webkit-box;
                        -webkit-line-clamp: 2;
                        -webkit-box-orient: vertical;
                        max-height: 2.4em;  /* 保证两行高度 */
                        text-overflow: ellipsis;
                        white-space: normal;
                        word-break: break-all;
                        '''
                    )
                    
                    # 文件大小
                    ui.label(f"{self.human_readable_size(file['size'])}").classes('text-sm text-gray-500 ml-2')
                    
                    # 删除按钮靠最右，防止换行挤到下面
                    ui.button(
                        '删除',
                        on_click=lambda i=idx: self.delete_file(i)
                    ).props('flat dense').classes('ml-auto')


    def disable_send_btn(self):
        self.send_btn.disable()
    def clear_pdf_list(self):
        self.pdf_list.clear()
    def insert_pdf_list(self,file_info):
        self.uploaded_files.append(file_info)  # 先添加新文件到列表
        self.update_file_list()  # 然后更新整个列表显示

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
                        self.status_var=content[0]
                    elif msg_type == "enable_buttons":
                        self.send_btn.enable()
                        self.add_btn.enable()
                    #add pdf
                    elif msg_type == "update_list":# need to add to pdf_paths before call this 
                        self.insert_pdf_list(content[0])
                    elif msg_type == "enable_button":
                        if content[0] == "add_btn":
                            self.add_btn.enable()
                        elif content[0] == "send_btn": 
                            self.send_btn.enable()
                    #delete pdf
                    elif msg_type == "remove_from_list":
                        self.pdf_list.default_slot.children[content[0]].delete()
            
                except self.parent.queue_Empty():
                    break
        finally:
            ui.timer(0.1, self.process_messages, once=True)
    def scroll_to_bottom(self):
        ui.run_javascript('''
            const pdflist = document.getElementById("pdf_list");
            if (pdflist) {
                pdflist.scrollTop = pdflist.scrollHeight;
            }
        ''')
    def update_status(self, message):
        self.status_var=message
    def selected_pdf_list_count(self):
        return [idx for idx, cb in self.selecte_items.items() if cb.value]
    def select_all_pdfs(self):
        """选择列表框中的所有PDF文件"""
        for checkbox in self.selecte_items.values():
            checkbox.value = True
        
        # 更新状态显示（可选）
        selected_count = len(self.selecte_items.items())
        self.update_status(f"已选择 {selected_count} 个PDF文件")
        # 可选：滚动到列表底部确保所有选项可见
        if selected_count > 0:
            self.scroll_to_bottom()

    def select_non_pdfs(self):
        """取消选择列表框中的PDF文件"""
        for checkbox in self.selecte_items.values():
            checkbox.value = False
        self.update_status(f"已取消选择 PDF文件")

    def clear_conversation(self):
        """清空对话历史"""
        self.conversation.clear()
        self.parent.conversation_history = []
        self.parent.message_queue.put(("系统", "对话历史已清空"))

    def delete_file(self,i):
        """从系统中移除选定的PDF文件
        
        1. 获取列表框中选中的文件
        2. 从各数据结构中移除对应项
        3. 重新合并剩余向量库
        4. 更新磁盘存储
        5. 刷新界面状态
        """
        #(target=self.parent._process_removal, args=(selected_indices,)
        file = self.uploaded_files[i]
        filepath = file['path']
        if os.path.exists(filepath):
            os.remove(filepath) # 删除上传的文件
        self.parent.remove_saved_individual_vec(filepath) #删除文件的向量
        del self.parent.individual_vectorstores[filepath] #个体向量文件
        self.parent.delete_vectors_by_metadata_path(filepath) #合并库内向量
        try:
            self.uploaded_files.remove(file) #子项目里去掉文件
        except Exception as e:
            print("not in the uploaded_files list, not newly uploaded")
        try:
            del self.parent.pdf_paths[i] #在主项目里去掉记录
            del self.parent.file_hashes[filepath] #在主项目里去掉hash
        except Exception as e:
            print("not in the pdf_paths list, program error, the pdf_paths is not synconized with the UI!")
            raise
        self.parent.save_vectorstore()
        self.update_file_list()
        ui.notify(f'已删除: {file["name"]}')

    def update_system_ready_message(self):
        """更新系统就绪消息"""
        model = self.model_var
        pdf_count = len(self.parent.pdf_paths)
        self.model_info_var=(f"当前模型: {model}")
        self.vectorstore_info=(f"向量库: 已加载 ({len(self.parent.individual_vectorstores)}个)")
        self.pdf_count_info=(f"已加载PDF: {pdf_count}个")
        self.last_update_info=(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        message = (
            f"RAG系统已准备好回答关于PDF内容的问题\n\n"
            f"当前模型: **{model}**\n"
            f"已加载PDF文档: **{pdf_count}**个\n"
            f"最后更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.parent.message_queue.put(("系统", message))
        self.send_btn.enable() # why

    def send_message(self, event=None):
        #没有向量库
        if not hasattr(self.parent, 'vectorstore') or self.parent.vectorstore is None:
            self.parent.message_queue.put(("message", "请先成功导入PDF文件或加载向量库"))
            return

        user_text = self.user_input.value.strip()
        #没有用户提示就不发送
        if not user_text:
            return
        #没有选定模型也不发送
        print(self.model_var)
        print(self.config_var)
        if not self.model_var.strip():
            self.parent.message_queue.put(("message", "请先选定模型"))
            return
        #if not hasattr(self, 'qa_chain') or self.qa_chain is None:
        if not self.get_conv_var():
            try:
                # 获取用户设置的系统提示
                system_prompt = self.prompt_input.value
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
                
                selected_indices = self.selected_pdf_list_count()
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
        else:#仅对话
            self.llm=self.parent.client.llmClient(
                backend=self.get_backend_var(),
                baseurl=self.get_config_var(),
                model=self.get_model_var()#,
                #temperature=self.localui.get_temp_var()
            )
        self.user_input.set_value("")
        #self.send_btn.disable()
        Thread(target=self.parent.get_answer, args=(user_text,), daemon=True).start()

    def get_config_var(self):
        return self.config_var
    def set_config_var(self,var):
        self.config_var_label.set_value(var)

    def get_backend_var(self):
        return self.backend_var
    def set_backend_var(self,var):
        self.backend_dropdown.set_value(var)
    
    def get_model_var(self):
        return self.model_var
    def set_model_var(self,var):
        self.model_dropdown.set_value(var)
    
    def get_temp_var(self):
        return self.temp_var
    def set_temp_var(self,var):
        self.temp_var_input.set_value(var)
        self.temp_var_slider.set_value(var)

    def get_check_var(self):
        return self.check_var
    def get_conv_var(self):
        return self.conv_var

    def get_system_prompt_var(self):
        return self.prompt_input.value
    def set_system_prompt_var(self,var):
        self.prompt_input.set_value(var)
    
    def get_max_tokens_var(self):
        return self.max_tokens_var
    def set_max_tokens_var(self,var):
        self.max_tokens_var_input.set_value(var)
        self.max_tokens_var_slider.set_value(var)

    #仅仅保留接口，web段不能处理文件夹，所以不做处理
    def get_pdf_folder_var(self):
        return self.pdf_folder_var
    def set_pdf_folder_var(self,var):
        self.pdf_folder_var=var

    def update_conversation(self, sender, message):
        """更新对话显示，支持Markdown格式和超链接"""
        with self.conversation:
            # 处理AI回复中的Markdown格式
            if sender == "AI":
                with ui.chat_message(name="AI", sent=False):
                    #self.insert_formatted_text(message)
                    ui.markdown(message).classes("w-full break-words")
            else:
                with ui.chat_message(name="User").classes("bg-blue-100 text-black rounded-lg shadow"):
                    ui.label(message).classes("whitespace-normal") 
                    #self.conversation.insert(tk.END, message)
        
            # 滚动到底部
            ui.run_javascript(f"""
                    var container = document.getElementById({self.conversation.id});
                    container.scrollTop = container.scrollHeight;
                """)
        # 保存到历史
        self.parent.conversation_history.append((sender, re.sub(r'<think>.*?</think>', '', message, flags=re.DOTALL)))

    def update_model_list(self, event=None):
        mbackend = self.backend_var
        self.ollama_models = self.parent.client.get_available_models(mbackend)
        self.parent.message_queue.put(("模型","self.ollama_models"))
        
        # 更新模型列表
        self.model_dropdown.set_options(self.ollama_models) 
        if mbackend == "ollama":
            self.config_var_label.set_value("http://localhost:11434")
        elif mbackend == "lmstudio":
            self.config_var_label.set_value("http://localhost:1234")
        elif mbackend == "openai":
            self.config_var_label.set_value("输入API Key")  # 清空用于输入API Key
        
        # 默认选择第一个模型
        if self.model_dropdown.options:
            self.model_dropdown.set_value(self.ollama_models[0])
            print (self.ollama_models[0])
        print (self.model_dropdown.options)

    def update_model(self, event):
        self.update_status(f"已选择模型: {self.get_model_var()}")

    def on_model_change(self, event):
        self.update_status(f"已选择后端: {self.get_backend_var()}")


if __name__ == "__main__":
    print("⚠️ 请不要直接运行此模块，正在跳转到主程序...")
    import sys
    from RAGv9 import main  # 导入主程序的 main 函数
    
    # 手动模拟 argparse
    if "--local" in sys.argv or "-l" in sys.argv:
        main(local_mode=True)  # 直接调用主函数
    else:
        main(local_mode=False)