import re
def preprocess_text(text: str) -> str:    
    text = re.sub(r"['\",\?:\-!,\;]", "", text)
    text = text.strip()
    text = " ".join(text.split())
    text = text.replace('\n',' ').strip().lower()
    return text