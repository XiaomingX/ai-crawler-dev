# 优化后网页爬虫API代码（单文件整合版）

## 一、依赖说明
先安装所需依赖，执行命令：
```bash
pip install fastapi uvicorn requests httpx pyppeteer beautifulsoup4 python-dotenv groq transformers boto3 pillow
```

## 二、.env文件配置（必须创建）
在代码同级目录创建`.env`文件，填入以下配置（根据实际情况修改）：
```env
# 认证配置
AUTH_SECRET=your_auth_secret_here

# Groq LLM配置
GROQ_API_KEY=your_groq_api_key_here
DETAIL_SYS_PROMPT=请总结网页内容，按段落清晰呈现
TAG_SELECTOR_SYS_PROMPT=根据提供的内容，从tag_list中选择最匹配的标签，用逗号分隔返回
LANGUAGE_SYS_PROMPT=将输入内容翻译成{language}，保持原意，语言自然
GROQ_MODEL=llama3-70b-8192
GROQ_MAX_TOKENS=5000

# S3/OSS存储配置
S3_ENDPOINT_URL=https://your-s3-endpoint.com
S3_ACCESS_KEY_ID=your_access_key
S3_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=your-bucket-name
S3_CUSTOM_DOMAIN=your-custom-domain.com（可选）
```


## 三、完整代码
```python
# 1. 基础依赖导入
import logging
import os
import random
import time
import re
from typing import List, Optional
from urllib.parse import urlparse
from io import BytesIO

# 第三方库导入
import httpx
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, BackgroundTasks, HTTPException
from pydantic import BaseModel
from pyppeteer import launch
from bs4 import BeautifulSoup
from groq import Groq
from transformers import LlamaTokenizer
import boto3
from botocore.client import Config
from PIL import Image

# 2. 全局初始化（只执行一次）
# 加载环境变量
load_dotenv()
# 配置日志（统一配置，避免重复）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# FastAPI实例
app = FastAPI(title="网页爬虫API", description="支持同步/异步网页爬取、内容处理与多语言转换")
# 全局认证密钥
SYSTEM_AUTH_SECRET = os.getenv('AUTH_SECRET')
# 随机用户代理列表
GLOBAL_AGENT_HEADERS = [
    "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14",
    "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Win64; x64; Trident/6.0)"
]


# 3. 工具类（简化封装，保留核心逻辑）
class CommonUtil:
    """通用工具类：URL处理、内容格式化"""
    @staticmethod
    def get_name_by_url(url: str) -> Optional[str]:
        """从URL提取域名和路径，生成唯一名称（替换特殊字符）"""
        if not url:
            return None
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")  # 移除www前缀
        path = parsed.path[:-1] if parsed.path.endswith("/") else parsed.path  # 移除末尾斜杠
        return (domain + path).replace("/", "-").replace(".", "-")

    @staticmethod
    def format_detail_content(detail: str) -> Optional[str]:
        """格式化LLM返回的详情内容：将**标题**转为### 标题"""
        if not detail:
            return None
        # 找到第一个#或*的位置，从该位置开始处理（跳过前面的无关内容）
        index1 = detail.find("#") if detail.find("#") != -1 else float('inf')
        index2 = detail.find("*") if detail.find("*") != -1 else float('inf')
        start_index = min(index1, index2) if min(index1, index2) != float('inf') else 0
        return re.sub(r'\*\*(.+?)\*\*', '### \\1', detail[start_index:])


class LLMUtil:
    """LLM工具类：内容总结、标签处理、多语言转换"""
    def __init__(self):
        # 初始化Groq客户端和Tokenizer
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.client = Groq(api_key=self.groq_api_key) if self.groq_api_key else None
        # LLM配置参数
        self.detail_sys_prompt = os.getenv('DETAIL_SYS_PROMPT')
        self.tag_sys_prompt = os.getenv('TAG_SELECTOR_SYS_PROMPT')
        self.lang_sys_prompt = os.getenv('LANGUAGE_SYS_PROMPT')
        self.groq_model = os.getenv('GROQ_MODEL', 'llama3-70b-8192')
        self.max_tokens = int(os.getenv('GROQ_MAX_TOKENS', 5000))
        # 初始化Tokenizer（用于文本长度截断）
        self.tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-65b")

    def _truncate_text(self, text: str) -> str:
        """截断文本到LLM最大token限制"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.max_tokens:
            return text
        logger.info(f"文本过长（{len(tokens)} tokens），截断到{self.max_tokens} tokens")
        return self.tokenizer.decode(tokens[:self.max_tokens])

    def call_llm(self, sys_prompt: str, user_prompt: str) -> Optional[str]:
        """核心LLM调用方法（统一封装，避免重复）"""
        # 校验必要参数
        if not (self.client and sys_prompt and user_prompt):
            logger.warning("LLM调用参数不完整，跳过调用")
            return None
        
        try:
            # 截断超长文本
            truncated_prompt = self._truncate_text(user_prompt)
            # 发起LLM请求
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": truncated_prompt}
                ],
                model=self.groq_model,
                temperature=0.2
            )
            return response.choices[0].message.content if response.choices else None
        except Exception as e:
            logger.error(f"LLM调用失败：{str(e)}")
            return None

    def process_detail(self, content: str) -> Optional[str]:
        """处理网页内容，生成结构化详情"""
        logger.info("开始处理网页详情")
        raw_detail = self.call_llm(self.detail_sys_prompt, content)
        return CommonUtil.format_detail_content(raw_detail)

    def process_tags(self, tags: List[str], content: str) -> List[str]:
        """根据内容筛选匹配的标签"""
        logger.info(f"开始处理标签（原始标签：{tags}）")
        user_prompt = f"tag_list is:{','.join(tags)}. content is: {content}"
        raw_result = self.call_llm(self.tag_sys_prompt, user_prompt)
        return [tag.strip() for tag in raw_result.split(',')] if raw_result else []

    def process_language(self, language: str, content: str) -> str:
        """将内容翻译为指定语言（英文直接返回，无需调用LLM）"""
        if 'english' in language.lower():
            return content
        
        logger.info(f"开始翻译内容到{language}")
        sys_prompt = self.lang_sys_prompt.replace("{language}", language)
        translated = self.call_llm(sys_prompt, content)
        if not translated:
            logger.warning(f"{language}翻译失败，返回原始内容")
            return content
        
        # 移除翻译结果中的Markdown标记（如###、**）
        return translated.replace("### ", "").replace("## ", "").replace("# ", "").replace("**", "")


class OSSUtil:
    """OSS工具类：文件上传、缩略图生成"""
    def __init__(self):
        # 初始化S3客户端
        self.endpoint = os.getenv('S3_ENDPOINT_URL')
        self.access_key = os.getenv('S3_ACCESS_KEY_ID')
        self.secret_key = os.getenv('S3_SECRET_ACCESS_KEY')
        self.bucket = os.getenv('S3_BUCKET_NAME')
        self.custom_domain = os.getenv('S3_CUSTOM_DOMAIN')
        
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4')
        ) if all([self.endpoint, self.access_key, self.secret_key, self.bucket]) else None

    def _get_file_url(self, file_key: str) -> Optional[str]:
        """生成文件访问URL（统一逻辑，避免重复）"""
        if not (self.s3_client and file_key):
            return None
        if self.custom_domain:
            return f"https://{self.custom_domain}/{file_key}"
        return f"{self.endpoint}/{self.bucket}/{file_key}"

    def _generate_file_key(self, url: str, is_thumbnail: bool = False) -> str:
        """生成文件存储Key（包含时间戳，避免重复）"""
        now = time.localtime()
        date_path = f"{now.tm_year}/{now.tm_mon}/{now.tm_mday}"
        base_name = CommonUtil.get_name_by_url(url) or str(random.randint(1000, 9999))
        if is_thumbnail:
            base_name += "-thumbnail"
        timestamp = int(time.time())
        return f"tools/{date_path}/{base_name}-{timestamp}.png"

    @staticmethod
    def compress_image(image_data: bytes) -> bytes:
        """压缩图片为WebP格式（减少存储和传输体积）"""
        with Image.open(BytesIO(image_data)) as img:
            buffer = BytesIO()
            img.save(buffer, format='WEBP', quality=85)
            return buffer.getvalue()

    def upload_file(self, file_path: str) -> Optional[str]:
        """上传本地文件到OSS（支持URL下载后上传）"""
        if not self.s3_client:
            logger.warning("OSS客户端未初始化，跳过文件上传")
            return None
        
        try:
            # 生成存储Key
            file_key = self._generate_file_key(file_path)
            # 处理文件来源（本地路径或URL）
            if file_path.startswith(('http://', 'https://')):
                # 从URL下载图片
                response = requests.get(file_path, headers={
                    'User-Agent': random.choice(GLOBAL_AGENT_HEADERS)
                })
                image_data = response.content
            else:
                # 读取本地文件
                with open(file_path, 'rb') as f:
                    image_data = f.read()
            
            # 压缩并上传
            compressed_data = self.compress_image(image_data)
            self.s3_client.upload_fileobj(
                BytesIO(compressed_data),
                self.bucket,
                file_key
            )
            
            # 删除本地临时文件（如果是本地路径）
            if os.path.exists(file_path):
                os.remove(file_path)
            
            file_url = self._get_file_url(file_key)
            logger.info(f"文件上传成功：{file_url}")
            return file_url
        except Exception as e:
            logger.error(f"文件上传失败：{str(e)}")
            return None

    def generate_thumbnail(self, image_url: str, original_url: str) -> Optional[str]:
        """生成图片缩略图（缩放为原尺寸的50%）"""
        if not (self.s3_client and image_url):
            logger.warning("生成缩略图参数不完整，跳过")
            return None
        
        try:
            # 从OSS下载原始图片
            parsed_url = urlparse(image_url)
            file_key = parsed_url.path.lstrip('/')  # 提取OSS中的文件Key
            response = self.s3_client.get_object(Bucket=self.bucket, Key=file_key)
            image_data = response['Body'].read()
            
            # 缩放图片
            with Image.open(BytesIO(image_data)) as img:
                new_size = (int(img.width * 0.5), int(img.height * 0.5))
                resized_img = img.resize(new_size)
            
            # 压缩并上传缩略图
            thumbnail_key = self._generate_file_key(original_url, is_thumbnail=True)
            buffer = BytesIO()
            resized_img.save(buffer, format='WEBP', quality=80)
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=thumbnail_key,
                Body=buffer.getvalue()
            )
            
            thumbnail_url = self._get_file_url(thumbnail_key)
            logger.info(f"缩略图生成成功：{thumbnail_url}")
            return thumbnail_url
        except Exception as e:
            logger.error(f"缩略图生成失败：{str(e)}")
            return None


class WebsiteCrawler:
    """网页爬虫类：页面加载、内容提取、截图"""
    def __init__(self):
        self.browser = None  # 浏览器实例（复用，减少启动开销）
        self.llm_util = LLMUtil()  # LLM工具实例
        self.oss_util = OSSUtil()  # OSS工具实例

    async def _init_browser(self):
        """初始化浏览器（懒加载，第一次使用时启动）"""
        if not self.browser:
            logger.info("启动浏览器实例")
            self.browser = await launch(
                headless=True,
                ignoreDefaultArgs=["--enable-automation"],
                ignoreHTTPSErrors=True,
                args=[
                    '--no-sandbox', '--disable-dev-shm-usage',
                    '--disable-gpu', '--disable-software-rasterizer'
                ]
            )

    async def scrape_website(self, url: str, tags: Optional[List[str]] = None, languages: Optional[List[str]] = None) -> Optional[dict]:
        """核心爬虫方法：加载页面、提取内容、处理数据"""
        start_time = int(time.time())
        logger.info(f"开始爬取网页：{url}")
        tags = tags or []
        languages = languages or []

        # 补全URL协议（如果缺少http/https）
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"

        try:
            # 1. 初始化浏览器并打开页面
            await self._init_browser()
            page = await self.browser.newPage()
            await page.setUserAgent(random.choice(GLOBAL_AGENT_HEADERS))
            await page.setViewport({'width': 1920, 'height': 1080})

            # 2. 加载页面（超时60秒，忽略加载超时错误）
            try:
                await page.goto(url, timeout=60000, waitUntil=['load', 'networkidle2'])
            except Exception as e:
                logger.warning(f"页面加载超时（{str(e)}），继续提取已加载内容")

            # 3. 提取页面基础信息
            html_content = await page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 标题（优先取<title>标签）
            title = soup.title.string.strip() if soup.title else ""
            # 描述（优先取meta description，其次取og:description）
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if not meta_desc:
                meta_desc = soup.find('meta', attrs={'property': 'og:description'})
            description = meta_desc['content'].strip() if meta_desc else ""
            # 完整文本内容
            full_content = soup.get_text().strip()

            # 4. 生成网页截图并上传OSS
            screenshot_path = f"./{CommonUtil.get_name_by_url(url)}.png"
            await page.screenshot({'path': screenshot_path, 'fullPage': True})
            screenshot_url = self.oss_util.upload_file(screenshot_path)
            # 生成缩略图
            thumbnail_url = self.oss_util.generate_thumbnail(screenshot_url, url) if screenshot_url else ""

            # 5. LLM处理：详情总结、标签筛选、多语言翻译
            detail = self.llm_util.process_detail(full_content)
            processed_tags = self.llm_util.process_tags(tags, full_content) if tags else []
            processed_langs = []
            for lang in languages:
                processed_langs.append({
                    'language': lang,
                    'title': self.llm_util.process_language(lang, title),
                    'description': self.llm_util.process_language(lang, description),
                    'detail': self.llm_util.process_language(lang, detail) if detail else ""
                })

            # 6. 整理返回结果
            result = {
                'name': CommonUtil.get_name_by_url(url),
                'url': url,
                'title': title,
                'description': description,
                'detail': detail,
                'screenshot_url': screenshot_url,
                'thumbnail_url': thumbnail_url,
                'tags': processed_tags,
                'languages': processed_langs
            }
            logger.info(f"网页爬取成功：{url}（耗时：{int(time.time()) - start_time}秒）")
            return result

        except Exception as e:
            logger.error(f"网页爬取失败：{url}，错误：{str(e)}")
            return None

        finally:
            # 关闭当前页面（保留浏览器实例复用）
            if 'page' in locals():
                await page.close()
            # 打印耗时
            logger.info(f"爬取流程结束：{url}（总耗时：{int(time.time()) - start_time}秒）")


# 4. Pydantic模型（请求参数校验）
class URLRequest(BaseModel):
    """同步爬取请求模型：基础参数"""
    url: str
    tags: Optional[List[str]] = None  # 可选：需要筛选的标签列表
    languages: Optional[List[str]] = None  # 可选：需要翻译的语言列表


class AsyncURLRequest(URLRequest):
    """异步爬取请求模型：新增回调参数"""
    callback_url: str  # 必须：处理完成后回调的URL
    callback_key: str  # 必须：回调请求的认证Key


# 5. 全局爬虫实例（避免重复初始化）
crawler = WebsiteCrawler()


# 6. API接口路由
@app.post('/site/crawl', summary="同步爬取网页")
async def sync_crawl(request: URLRequest, authorization: Optional[str] = Header(None)):
    """
    同步爬取网页：
    - 实时返回爬取结果
    - 支持标签筛选和多语言翻译
    - 需要认证时，请求头需携带 Authorization: Bearer {AUTH_SECRET}
    """
    # 认证校验（如果配置了AUTH_SECRET）
    if SYSTEM_AUTH_SECRET:
        if not authorization or authorization != f"Bearer {SYSTEM_AUTH_SECRET}":
            raise HTTPException(status_code=401, detail="认证失败：Authorization头错误或缺失")
    
    # 执行爬取
    result = await crawler.scrape_website(request.url, request.tags, request.languages)
    
    # 整理响应
    return {
        'code': 200 if result else 10001,
        'msg': 'success' if result else 'fail: 处理异常，请稍后重试',
        'data': result
    }


@app.post('/site/crawl_async', summary="异步爬取网页")
async def async_crawl(
    background_tasks: BackgroundTasks,
    request: AsyncURLRequest,
    authorization: Optional[str] = Header(None)
):
    """
    异步爬取网页：
    - 立即返回任务状态，后台执行爬取
    - 爬取完成后调用 callback_url 回调结果
    - 回调请求头携带 Authorization: Bearer {callback_key}
    """
    # 认证校验
    if SYSTEM_AUTH_SECRET:
        if not authorization or authorization != f"Bearer {SYSTEM_AUTH_SECRET}":
            raise HTTPException(status_code=401, detail="认证失败：Authorization头错误或缺失")
    
    # 将爬取任务添加到后台（不阻塞响应）
    background_tasks.add_task(
        _async_crawl_worker,
        url=request.url,
        tags=request.tags,
        languages=request.languages,
        callback_url=request.callback_url,
        callback_key=request.callback_key
    )
    
    # 立即返回任务已接受
    return {'code': 200, 'msg': 'success: 爬取任务已启动，完成后将回调通知'}


async def _async_crawl_worker(url: str, tags: List[str], languages: List[str], callback_url: str, callback_key: str):
    """异步爬取工作器：后台执行爬取，完成后回调"""
    # 执行爬取
    result = await crawler.scrape_website(url, tags, languages)
    
    # 回调通知（使用异步HTTP客户端，避免阻塞）
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                url=callback_url,
                json={
                    'task_status': 'success' if result else 'fail',
                    'target_url': url,
                    'result': result
                },
                headers={'Authorization': f"Bearer {callback_key}"},
                timeout=30
            )
            logger.info(f"异步任务回调成功：{callback_url}（目标URL：{url}）")
        except Exception as e:
            logger.error(f"异步任务回调失败：{callback_url}，错误：{str(e)}")


# 7. 启动入口
if __name__ == '__main__':
    import uvicorn
    # 启动服务（默认端口8040，支持所有IP访问）
    logger.info("启动FastAPI服务：http://0.0.0.0:8040/docs")
    uvicorn.run(app, host="0.0.0.0", port=8040)
```


## 四、使用说明
1. **启动服务**：直接运行代码，服务会启动在 `http://0.0.0.0:8040`，可通过 `http://localhost:8040/docs` 查看API文档（自动生成）。
2. **同步爬取**：调用 `/site/crawl` 接口，传入 `url` 等参数，实时获取爬取结果。
3. **异步爬取**：调用 `/site/crawl_async` 接口，传入 `callback_url` 和 `callback_key`，服务后台执行，完成后会向 `callback_url` 发送POST请求返回结果。


要不要我帮你整理一份**API调用示例文档**？包含Postman请求示例、回调接收代码示例，方便你直接测试使用。