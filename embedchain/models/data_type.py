from enum import Enum

class DataType(str, Enum):
    web_page = "web_page"
    file = "file"
    directory = "directory"
    youtube_video = "youtube_video"
    document = "document"
    code = "code"
    image = "image"
    audio = "audio"
    text = "text"
