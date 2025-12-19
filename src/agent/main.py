"""
LangChain1.0多模态RAG智能问答系统 - 完整实现（含PDF解析与溯源）
"""

import json
import base64
import uvicorn
import re
import os
import io
import fitz  # PyMuPDF
from PIL import Image

from typing import List, Dict, Any, AsyncGenerator, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import HTTPException, FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==================== 数据结构定义 ====================
class ContentBlock(BaseModel):
    type: str = Field(description="内容类型: text, image, audio")
    content: Optional[str] = Field(description="内容数据")


class MessageRequest(BaseModel):
    content_blocks: List[ContentBlock] = Field(default=[], description="内容块")
    history: List[Dict[str, Any]] = Field(default=[], description="对话历史")
    pdf_chunks: List[Dict[str, Any]] = Field(default=[], description="PDF文档块信息，用于引用溯源")


class MessageResponse(BaseModel):
    content: str
    timestamp: str
    role: str
    references: List[Dict[str, Any]] = Field(default_factory=list)


# ==================== 工具类 ====================
class ImageProcessor:
    """图像处理工具类"""

    @staticmethod
    def image_to_base64(image_file: UploadFile) -> str:
        try:
            contents = image_file.file.read()
            base64_encoded = base64.b64encode(contents).decode('utf-8')
            return base64_encoded
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"图像编码失败: {str(e)}")

    @staticmethod
    def get_image_mime_type(filename: str) -> str:
        extension = filename.split('.')[-1].lower()
        mime_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
            'webp': 'image/webp'
        }
        return mime_types.get(extension, 'image/jpeg')


class AudioProcessor:
    """音频处理工具类"""

    @staticmethod
    def audio_to_base64(audio_file: UploadFile) -> str:
        try:
            if not AudioProcessor.is_valid_audio_type(audio_file.content_type, audio_file.filename):
                raise HTTPException(
                    status_code=400,
                    detail="不支持的音频格式，支持的格式有: MP3, WAV, OGG, M4A, FLAC"
                )

            contents = audio_file.file.read()
            max_size = 10 * 1024 * 1024
            if len(contents) > max_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"音频文件过大，最大支持 {max_size // 1024 // 1024}MB"
                )

            base64_encoded = base64.b64encode(contents).decode('utf-8')
            return base64_encoded

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"音频编码失败: {str(e)}")

    @staticmethod
    def get_audio_mime_type(filename: str) -> str:
        extension = filename.split('.')[-1].lower()
        mime_types = {
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'm4a': 'audio/mp4',
        }
        return mime_types.get(extension, 'audio/mpeg')

    @staticmethod
    def is_valid_audio_type(content_type: str, filename: str) -> bool:
        supported_mimes = {'audio/mpeg', 'audio/wav', 'audio/mp4'}
        if content_type and content_type in supported_mimes:
            return True

        file_extension = filename.split('.')[-1].lower()
        supported_extensions = {'mp3', 'wav', 'm4a'}
        return file_extension in supported_extensions


class PDFProcessor:
    """PDF处理工具类"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    async def extract_pdf_pages_as_images(self, file_content: bytes, max_pages: int = 5) -> List[str]:
        """将PDF页面提取为Base64图像（用于OCR处理）"""
        try:
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            total_pages = len(pdf_document)
            pages_to_extract = min(max_pages, total_pages)

            images = []
            for page_num in range(pages_to_extract):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                images.append(base64_image)

            pdf_document.close()
            return images
        except Exception as e:
            raise Exception(f"PDF页面提取失败: {str(e)}")

    async def process_pdf(self, file_content: bytes, filename: str):
        """流式处理PDF文档，返回处理进度和结果"""
        try:
            # 保存临时文件
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            tmp_file_path = os.path.join(temp_dir, filename)

            with open(tmp_file_path, 'wb') as f:
                f.write(file_content)

            full_text = ""
            doc = fitz.open(tmp_file_path)
            pages_content = {}

            # 逐页读取内容
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                full_text += text
                pages_content[page_num + 1] = text

            print(f"合并后文本长度: {len(full_text)} 字符")
            preview = full_text[:200] if full_text else "空内容"
            print(f"文本预览: {repr(preview)}")

            # 使用RecursiveCharacterTextSplitter进行智能分块
            text_chunks = self.text_splitter.split_text(full_text)
            print(f"文本分块完成，共 {len(text_chunks)} 个块")

            # 构建带元数据的文档块
            document_chunks = []
            for i, chunk in enumerate(text_chunks):
                if chunk.strip():  # 过滤空块
                    page_number = 1
                    sorted_keys = sorted(pages_content.keys())
                    for page_num in sorted_keys:
                        if chunk.strip()[:50] in pages_content[page_num]:
                            page_number = page_num
                            break

                    doc_chunk = {
                        "id": f"{filename}_{i}",
                        "content": chunk.strip(),
                        "metadata": {
                            "source": filename,
                            "chunk_id": i,
                            "chunk_size": len(chunk),
                            "total_chunks": len(text_chunks),
                            "page_number": page_number,
                            "reference_id": f"[{i}]",
                            "source_info": f"{filename} - 第{page_number}页"
                        }
                    }
                    document_chunks.append(doc_chunk)

            print(f"处理完成！共生成 {len(document_chunks)} 个文档块")

            # 清理临时文件
            os.remove(tmp_file_path)

            return document_chunks
        except Exception as e:
            print(f"PDF处理失败: {str(e)}")
            raise Exception(f"PDF处理失败: {str(e)}")


# ==================== 模型初始化 ====================
def get_chat_model():
    """初始化多模态模型"""
    try:
        model = init_chat_model(
            model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
            model_provider="openai",
            base_url="https://api.siliconflow.cn/v1/",
            api_key="你的硅基流动api_key",  # 请替换为实际的API Key
        )
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型初始化失败: {str(e)}")


# ==================== 消息处理函数 ====================
def create_multimodal_message(request: MessageRequest, image_file: UploadFile | None = None, audio_file: UploadFile | None = None) -> HumanMessage:
    """创建多模态消息"""
    message_content = []

    # 如果有图片
    if image_file:
        processor = ImageProcessor()
        mime_type = processor.get_image_mime_type(image_file.filename)
        base64_image = processor.image_to_base64(image_file)
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}"
            },
        })

    # 如果有音频
    if audio_file:
        processor = AudioProcessor()
        mime_type = processor.get_audio_mime_type(audio_file.filename)
        base64_audio = processor.audio_to_base64(audio_file)
        message_content.append({
            "type": "audio_url",
            "audio_url": {
                "url": f"data:{mime_type};base64,{base64_audio}"
            },
        })

    # 处理内容块
    for block in request.content_blocks:
        if block.type == "text" and block.content:
            message_content.append({
                "type": "text",
                "text": block.content
            })
        elif block.type == "image" and block.content:
            if block.content.startswith("data:image"):
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": block.content
                    },
                })
        elif block.type == "audio" and block.content:
            if block.content.startswith("data:audio"):
                message_content.append({
                    "type": "audio_url",
                    "audio_url": {
                        "url": block.content
                    },
                })

    # 处理PDF文档块
    if request.pdf_chunks:
        pdf_content = "\n\n=== 参考文档内容 ===\n"
        for i, chunk in enumerate(request.pdf_chunks):
            content = chunk.get("content", "")
            source_info = chunk.get("metadata", {}).get(
                "source_info", f"文档块 {i}")
            pdf_content += f"\n[{i}] {content}\n来源: {source_info}\n"
        pdf_content += "\n请在回答时引用相关内容，使用格式如 [0]、[1] 等。\n"

        # 将PDF内容附加到最后一个文本块
        for i in range(len(message_content) - 1, -1, -1):
            item = message_content[i]
            if item['type'] == 'text':
                item['text'] += pdf_content
                break
        else:
            # 如果没有文本块，创建一个新的
            message_content.append({
                "type": "text",
                "text": pdf_content
            })

    if not message_content:
        raise ValueError("消息内容不能为空")

    return HumanMessage(content=message_content)


def convert_history_to_messages(history: List[Dict[str, Any]]) -> List[BaseMessage]:
    """将历史记录转换为 LangChain 消息格式，支持多模态内容"""
    messages = []

    # 添加系统消息
    system_prompt = """
        你是一个专业的多模态 RAG 助手，具备如下能力：
        1. 与用户对话的能力。
        2. 图像内容识别和分析能力(OCR, 对象检测，场景理解)
        3. 音频转写与分析
        4. 知识检索与问答
        
        重要指导原则：
        - 当用户上传图片并提出问题时，请结合图片内容和用户的具体问题来回答
        - 仔细分析图片中的文字、图表、对象、场景等所有可见信息
        - 根据用户的问题重点，有针对性地分析图片相关部分
        - 如果图片包含文字，请准确识别并在回答中引用
        - 如果用户只上传图片没有问题，则提供图片的全面分析
        
        引用格式要求（重要）：
        - 当回答基于提供的参考文档内容时，必须在相关信息后添加引用标记，格式为[0]、[1]等
        - 引用标记应紧跟在相关内容后面，如："这是重要信息[0]"
        - 每个不同的文档块使用对应的引用编号
        - 如果用户消息中包含"=== 参考文档内容 ==="部分，必须使用其中的内容来回答问题并添加引用
        - 只需要在正文中使用角标引用，不需要在最后列出"参考来源"
        
        请以专业、准确、友好的方式回答，并严格遵循引用格式。当有参考文档时，优先使用文档内容回答。
    """
    messages.append(SystemMessage(content=system_prompt))

    # 转换历史消息
    for msg in history:
        content = msg.get("content", "")
        content_blocks = msg.get("content_blocks", [])
        message_content = []

        if msg["role"] == "user":
            for block in content_blocks:
                if block.get("type") == "text":
                    message_content.append({
                        "type": "text",
                        "text": block.get("content", "")
                    })
                elif block.get("type") == "image":
                    image_data = block.get("content", "")
                    if image_data.startswith("data:image"):
                        message_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": image_data
                            }
                        })
                elif block.get("type") == "audio":
                    audio_data = block.get("content", "")
                    if audio_data.startswith("data:audio"):
                        message_content.append({
                            "type": "audio_url",
                            "audio_url": {
                                "url": audio_data
                            }
                        })

            if message_content:
                messages.append(HumanMessage(content=message_content))
            elif content:
                messages.append(HumanMessage(content=content))

        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=content))

    return messages


def extract_references_from_content(content: str, pdf_chunks: list = None) -> list:
    """从模型回答中提取引用信息"""
    references = []

    if not content or not pdf_chunks:
        return references

    print('模型输出内容:', content[:500])

    # 匹配格式为 [0], [1] 等的引用标记
    reference_pattern = r'\[(\d+)\]'
    matches = re.findall(reference_pattern, content)
    print(f"找到的引用标记: {matches}")

    if matches:
        for match in matches:
            try:
                ref_num = int(match)
                if 0 <= ref_num < len(pdf_chunks):
                    chunk = pdf_chunks[ref_num]
                    reference = {
                        "id": ref_num,
                        "text": chunk.get("content", "")[:200] + "..." if len(
                            chunk.get("content", "")) > 200 else chunk.get("content", ""),
                        "source": chunk.get("metadata", {}).get("source", "未知来源"),
                        "page": chunk.get("metadata", {}).get("page_number", 1),
                        "chunk_id": chunk.get("metadata", {}).get("chunk_id", 0),
                        "source_info": chunk.get("metadata", {}).get("source_info", "未知来源")
                    }
                    references.append(reference)
            except ValueError:
                continue

    return references


# ==================== 流式响应生成 ====================
async def generate_streaming_response(
        messages: List[BaseMessage],
        pdf_chunks: List[Dict[str, Any]] = None
) -> AsyncGenerator[str, None]:
    """生成流式响应"""
    try:
        model = get_chat_model()
        full_response = ""
        chunk_count = 0

        async for chunk in model.astream(messages):
            chunk_count += 1
            if hasattr(chunk, 'content') and chunk.content:
                content = chunk.content
                full_response += content

                data = {
                    "type": "content_delta",
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        # 提取引用信息
        references = extract_references_from_content(full_response, pdf_chunks)

        # 发送完成信号
        final_data = {
            "type": "message_complete",
            "full_content": full_response,
            "timestamp": datetime.now().isoformat(),
            "references": references
        }
        yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"

    except Exception as e:
        error_data = {
            "type": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"


# ==================== API接口函数 ====================
async def chat_stream(request: MessageRequest, image_file: UploadFile | None = None,
                      audio_file: UploadFile | None = None, pdf_file: UploadFile | None = None):
    """流式聊天接口（支持多模态）"""
    try:
        # 转换消息历史
        messages = convert_history_to_messages(request.history)

        # 如果有PDF文件，先处理PDF
        pdf_chunks = None
        if pdf_file:
            pdf_processor = PDFProcessor()
            pdf_content = await pdf_file.read()
            pdf_chunks = await pdf_processor.process_pdf(file_content=pdf_content, filename=pdf_file.filename)
            request.pdf_chunks = pdf_chunks

        # 添加当前用户消息（支持多模态）
        current_message = create_multimodal_message(request, image_file, audio_file)
        messages.append(current_message)

        # 返回流式响应
        return StreamingResponse(
            generate_streaming_response(messages, pdf_chunks),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def chat_sync(request: MessageRequest):
    """同步聊天接口（支持多模态）"""
    try:
        messages = convert_history_to_messages(request.history)
        current_message = create_multimodal_message(request)
        messages.append(current_message)

        model = get_chat_model()
        response = await model.ainvoke(messages)

        # 提取引用信息
        references = extract_references_from_content(response.content, request.pdf_chunks)

        return MessageResponse(
            content=response.content,
            role="assistant",
            timestamp=datetime.now().isoformat(),
            references=references
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== FastAPI应用 ====================
app = FastAPI(
    title="多模态 RAG 工作台 API",
    description="基于 LangChain 1.0 的智能对话 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 配置跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ==================== API路由 ====================
@app.post("/api/chat/stream", response_class=StreamingResponse)
async def api_chat_stream(
    image_file: UploadFile | None = File(None),
    audio_file: UploadFile | None = File(None),
    pdf_file: UploadFile | None = File(None),
    content_blocks: str = Form(default="[]"),
    history: str = Form(default="[]")
):
    """流式聊天API端点（支持图片、音频和PDF上传）"""
    try:
        # 解析 JSON 字符串
        try:
            content_blocks_data = json.loads(content_blocks)
            history_data = json.loads(history)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"JSON 解析错误: {str(e)}")

        # 创建请求对象
        request_data = MessageRequest(
            content_blocks=[ContentBlock(**block) for block in content_blocks_data],
            history=history_data
        )

        return await chat_stream(request_data, image_file, audio_file, pdf_file)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=MessageResponse)
async def api_chat_sync(request: MessageRequest):
    """同步聊天API端点"""
    return await chat_sync(request)


@app.post("/api/chat/simple", response_model=MessageResponse)
async def api_chat_simple(
    message: str = Form(...),
    history: str = Form(default="[]")
):
    """简化的聊天API端点（仅支持文本）"""
    try:
        history_data = json.loads(history) if history else []
        request_data = MessageRequest(
            content_blocks=[ContentBlock(type="text", content=message)],
            history=history_data
        )
        return await chat_sync(request_data)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON 解析错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process_pdf")
async def api_process_pdf(pdf_file: UploadFile = File(...)):
    """独立处理PDF文件的API端点"""
    try:
        pdf_processor = PDFProcessor()
        pdf_content = await pdf_file.read()
        chunks = await pdf_processor.process_pdf(file_content=pdf_content, filename=pdf_file.filename)

        return {
            "status": "success",
            "filename": pdf_file.filename,
            "chunks_count": len(chunks),
            "chunks": chunks[:5],  # 只返回前5个块作为预览
            "total_size": len(pdf_content)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "多模态 RAG 工作台 API",
        "version": "1.0.0",
        "endpoints": {
            "流式聊天（支持图片/音频/PDF）": "POST /api/chat/stream",
            "同步聊天": "POST /api/chat",
            "简化聊天（仅文本）": "POST /api/chat/simple",
            "独立处理PDF": "POST /api/process_pdf",
            "API文档": "GET /docs"
        },
        "支持的文件格式": {
            "图片": "jpg, jpeg, png, gif, bmp, webp",
            "音频": "mp3, wav, m4a",
            "PDF": "所有可读的PDF格式"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("多模态 RAG 工作台 API 服务")
    print("=" * 60)
    print(f"API文档地址: http://localhost:8000/docs")
    print(f"流式聊天接口: POST http://localhost:8000/api/chat/stream")
    print(f"同步聊天接口: POST http://localhost:8000/api/chat")
    print(f"独立PDF处理: POST http://localhost:8000/api/process_pdf")
    print("=" * 60)

    # 确保临时目录存在
    os.makedirs("temp", exist_ok=True)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
    )