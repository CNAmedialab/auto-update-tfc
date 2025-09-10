
import json
import re
import os
import random
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
import sys
import time
from datetime import datetime
from es_Upload import ESUploader

load_dotenv()

# 爬蟲歷史記錄文件
CRAWLER_HISTORY_FILE = 'tfc_crawler_history.json'

def load_crawler_history():
    """載入爬蟲歷史記錄"""
    if os.path.exists(CRAWLER_HISTORY_FILE):
        try:
            with open(CRAWLER_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_crawler_history(history):
    """儲存爬蟲歷史記錄"""
    with open(CRAWLER_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def update_last_crawled_title(title):
    """更新最後爬取的文章標題"""
    history = load_crawler_history()
    history['last_crawled_title'] = title
    history['last_crawled_time'] = datetime.now().isoformat()
    save_crawler_history(history)

def get_last_crawled_title():
    """獲取最後爬取的文章標題"""
    history = load_crawler_history()
    return history.get('last_crawled_title', None)

## functions ##
def text_embeddings_3(text):
    client = OpenAI()
    
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    tokens = encoding.encode(text)
    
    max_tokens = 8000 
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        text = encoding.decode(truncated_tokens)
        print(f">>> 文本過長 ({len(tokens)} tokens)，已截斷至 {max_tokens} tokens")
    
    t = client.embeddings.create(model="text-embedding-3-large", input=text)
    return t.data[0].embedding

# 存入elasticsearch - 使用 ESUploader 類
def save_to_es(data, max_retries=3, retry_delay=2):
    """
    優化的 ES 上傳函數，具有重試機制和更好的錯誤處理
    :param data: 文章資料
    :param max_retries: 最大重試次數
    :param retry_delay: 重試間隔（秒）
    """
    import time
    
    for attempt in range(max_retries):
        try:
            # 創建 ESUploader 實例
            uploader = ESUploader()
            
            # 設定索引名稱
            index_name = "lab_tfc_search_test"
            
            # 資料驗證和清理
            if not data.get('title', '').strip():
                return {
                    "Result": "N",
                    "Message": "文章標題不能為空"
                }
            
            # 處理空日期問題
            if not data.get('date', '').strip():
                from datetime import datetime
                data['date'] = datetime.now().strftime("%Y-%m-%d")
                print(f"    使用當前日期: {data['date']}")
            
            # 確保 pid 存在
            if 'pid' not in data or not data['pid']:
                import re
                from datetime import datetime
                
                date_str = data.get('date', '')
                date_prefix = ''
                
                if date_str:
                    date_pattern = r'(\d{4})-(\d{1,2})-(\d{1,2})'
                    match = re.search(date_pattern, date_str)
                    if match:
                        year, month, day = match.groups()
                        date_prefix = f"{year}{month.zfill(2)}{day.zfill(2)}"
                
                if not date_prefix:
                    try:
                        data_date = datetime.strptime(data['date'], "%Y-%m-%d")
                        date_prefix = data_date.strftime("%Y%m%d")
                    except:
                        date_prefix = datetime.now().strftime("%Y%m%d")
                
                # 使用 article_id 或 hash
                article_id = data.get('article_id', '')
                if article_id:
                    data['pid'] = f"{date_prefix}{article_id}"
                else:
                    import hashlib
                    title = data.get('title', '')
                    title_hash = hashlib.md5(title.encode('utf-8')).hexdigest()
                    serial_num = str(int(title_hash[-4:], 16) % 1000).zfill(3)
                    data['pid'] = f"{date_prefix}{serial_num}"
            
            # 快速檢查標題是否已存在（使用更高效的查詢）
            title = data.get('title', '')
            if title:
                try:
                    search_query = {
                        "query": {
                            "bool": {
                                "must": [
                                    {"match_phrase": {"title": title}}
                                ]
                            }
                        },
                        "size": 1,
                        "_source": ["title", "pid"]  # 只取需要的欄位
                    }
                    
                    es = uploader.es_client
                    search_result = es.search(
                        index=index_name, 
                        timeout='5s',  # 設定超時
                        **search_query
                    )
                    
                    if search_result['hits']['total']['value'] > 0:
                        existing_id = search_result['hits']['hits'][0]['_id']
                        print(f"    標題已存在，跳過 (現有ID: {existing_id})")
                        return {
                            "Result": "SKIP",
                            "Message": f"Document with same title already exists",
                            "existing_id": existing_id
                        }
                        
                except Exception as search_error:
                    print(f"    搜尋檢查時出錯 (嘗試 {attempt + 1}/{max_retries}): {search_error}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        print("    搜尋失敗，繼續上傳流程")
            
            # 清理資料中的無效欄位
            cleaned_data = {}
            for key, value in data.items():
                if value is not None and value != "":
                    cleaned_data[key] = value
                else:
                    cleaned_data[key] = ""  # ES 不接受 None 值
            
            # 上傳到 ES
            result = uploader.upload_single_document(
                document=cleaned_data,
                index=index_name,
                strict_fields=False
            )
            
            if result and result.get("Result") == "Y":
                print(f"    上傳成功 (文檔ID: {result.get('ResultData', {}).get('document_id', 'unknown')})")
                return result
            else:
                error_msg = result.get('Message', 'Unknown error') if result else 'No response'
                print(f"    上傳失敗 (嘗試 {attempt + 1}/{max_retries}): {error_msg}")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return {
                        "Result": "N",
                        "Message": f"Upload failed after {max_retries} attempts: {error_msg}"
                    }
                    
        except Exception as e:
            print(f"    ES操作出錯 (嘗試 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                return {
                    "Result": "N",
                    "Message": f"ES operation failed after {max_retries} attempts: {str(e)}"
                }
    
    return {
        "Result": "N",
        "Message": "Unexpected error in save_to_es"
    }
        
def backup_to_jsonl(report, backup_file='report_uploaded.jsonl'):
    """
    備份單筆資料到 JSONL 文件
    :param report: 報告資料 (dict)
    :param backup_file: 備份文件名
    """
    try:
        with open(backup_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(report, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        print(f"    ⚠️ 備份失敗: {e}")
        return False

def embedding_and_save(report):
    """
    處理單篇文章：text_embeddings_3 -> 備份 -> save_to_es
    :param report: 來自tfc_crawler的文章資料
    :return: 處理結果字典
    """
    try:
        print(f">>> 處理文章：{report['title']}")
        
        # 1. Embedding
        if report.get('full_content', '').strip():
            report['embeddings'] = text_embeddings_3(report['full_content'])
            print("    ✅ Embedding完成")
        else:
            report['embeddings'] = []
        
        # 2. 備份
        backup_to_jsonl(report)
        
        # 3. 上傳ES
        result = save_to_es(report)
        
        if result and result.get('Result') == 'Y':
            return {'status': 'success', 'backup': True}
        elif result and result.get('Result') == 'SKIP':
            return {'status': 'skip', 'backup': True}
        else:
            return {'status': 'error', 'backup': True}
            
    except Exception as e:
        print(f"    ❌ 處理出錯: {e}")
        return {'status': 'error', 'backup': False}

def main_process(max_pages=40):
    """
    爬蟲 -> embedding_and_save -> save to jsonl and es
    :param max_pages: 最大爬取頁數
    """
    
    # 統計資料
    total_processed = 0
    success_count = 0
    skip_count = 0 
    error_count = 0
    backup_count = 0
    first_article_title = None  # 記錄第一篇（最新）文章標題
    
    try:
        # 即時處理：使用tfc_crawler的爬蟲邏輯，但每爬到一篇立即處理
        base_url = "https://tfc-taiwan.org.tw/fact-check-reports-all/"
        page = 1
        
        # 獲取上次爬取的最後一篇文章標題
        last_crawled_title = get_last_crawled_title()
        if last_crawled_title:
            print(f">>> 上次爬取的最後一篇文章：{last_crawled_title}")
        
        duplicate_found = False
        
        while not duplicate_found and page <= max_pages:
            # 構建當前頁面的URL
            current_url = base_url if page == 1 else f"{base_url}?pg={page}"
            
            print(f"\n>>> 正在爬取第 {page} 頁: {current_url}")
            
            response = requests.get(current_url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.select('li.kb-query-item')
            
            if not articles:
                print(f"第 {page} 頁沒有找到文章，停止爬取")
                break
                
            print(f"第 {page} 頁找到 {len(articles)} 篇文章")
            
            # 找到所有 fact-check-reporter-{id}
            reporter_articles = []
            for i, article in enumerate(articles):
                article_classes = article.get('class', [])
                for cls in article_classes:
                    if cls.startswith('fact-check-reporter-'):
                        article_id = cls.replace('fact-check-reporter-', '')
                        reporter_articles.append({
                            'index': i,
                            'article': article,
                            'article_id': article_id
                        })
                        break
            
            print(f"在第 {page} 頁找到 {len(reporter_articles)} 個report list")
            
            # 即時處理每篇文章
            for article_info in reporter_articles:
                if duplicate_found:
                    break
                    
                try:
                    i = article_info['index']
                    article = article_info['article']
                    article_id = article_info['article_id']
                    
                    print(f"\n=== 第{page}頁第{i+1}篇文章 (ID: {article_id}) ===")
                    
                    # 提取標題
                    title_elem = article.select_one('div.kb-dynamic-html')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    if not title:
                        continue
                        
                    print(f">>> 標題: {title}")
                    
                    # 檢查是否遇到重複文章
                    duplicate = False
                    try:
                        with open('report_uploaded.jsonl', 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    record = json.loads(line)
                                    if record.get('title') == title:
                                        duplicate = True
                                        break
                                except Exception:
                                    continue
                    except FileNotFoundError:
                        pass

                    if duplicate:
                        print(f">>> 遇到重複文章，跳過：{title}")
                        continue
                    elif last_crawled_title and title == last_crawled_title:
                        print(">>> 停止爬取")
                        duplicate_found = True
                        break
                    
                    # 使用tfc_crawler的邏輯提取資料
                    # 提取摘要
                    summary_elem = article.select_one('div[class*="kb-dynamic-html-id-"][class*="_c9ad79-23"]')
                    summary = summary_elem.get_text(strip=True)[:200] if summary_elem else ""
                    
                    # 提取 label 和 category
                    label = ""
                    category = ""
                    dynamic_lists = article.select('ul.wp-block-kadence-dynamiclist')
                    
                    for dynamic_list in dynamic_lists:
                        link_elem = dynamic_list.select_one('a.kb-dynamic-list-item-link')
                        if link_elem:
                            href = link_elem.get('href', '')
                            text = link_elem.get_text(strip=True)
                            
                            if 'fact-check-report-classification/' in href:
                                label = text
                                print(f">>> label: {label}")
                            elif 'fact-check-report-type/' in href:
                                category = text
                                print(f">>> category: {category}")
                    
                    # 提取日期
                    date = ""
                    date_elem = article.select_one('div[class*="kt-adv-heading"]')
                    if date_elem:
                        date_text = date_elem.get_text(strip=True)
                        date_match = re.search(r'(\d{4}-\d{1,2}-\d{1,2})', date_text)
                        if date_match:
                            date = date_match.group(1)
                    
                    # 提取文章連結
                    link = ""
                    link_elem = article.select_one('a.kb-button')
                    if link_elem and link_elem.get('href'):
                        link = link_elem['href']
                    
                    # 生成 pid
                    if article_id == 'fact-check-reporter':
                        random_num = str(random.randint(300, 999))
                        pid = f"{date.replace('-', '')}{random_num}"
                        print(f">>> pid: {pid}")
                    else:
                        pid = f"{date.replace('-', '')}{article_id}"
                        print(f">>> pid: {pid}")
                        
                    # 建立基本報告資料
                    report = {
                        'title': title,
                        'link': link,
                        'date': date,
                        'label': label,
                        'category': category,
                        'summary': summary,
                        'full_content': "",
                        'pid': pid
                    }

                    # 如果有連結，抓取完整內容（使用tfc_crawler的完整內容邏輯）
                    if link:
                        try:
                            print(">>> 開始抓取完整內容")
                            article_response = requests.get(link, headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                            })
                            article_response.raise_for_status()
                            
                            article_soup = BeautifulSoup(article_response.content, 'html.parser')
                            content_container = article_soup.select_one('article')
                            
                            if not content_container:
                                content_container = article_soup.select_one('div.entry-content, div.post-content')
                            
                            if content_container:
                                # 移除圖片和圖片說明
                                for img_elem in content_container.find_all(['img', 'figure', 'figcaption']):
                                    img_elem.decompose()
                                
                                # 再一次尋找 label (如果之前沒找到的話)
                                if not label or label == "":
                                    # 在文章詳細頁面尋找 label 的動態清單
                                    detail_dynamic_lists = content_container.select('ul.wp-block-kadence-dynamiclist')
                                    for dynamic_list in detail_dynamic_lists:
                                        link_elem = dynamic_list.select_one('a.kb-dynamic-list-item-link')
                                        if link_elem:
                                            href = link_elem.get('href', '')
                                            text = link_elem.get_text(strip=True)
                                            if 'fact-check-report-classification/' in href:
                                                label = text
                                                report['label'] = label  # 更新 report 中的 label
                                                print(f">>> label: {label}")
                                                break
                                
                                # 提取完整摘要
                                summary_elems = content_container.select('p[class*="kt-adv-heading"]')
                                if summary_elems:
                                    for elem in summary_elems:
                                        text = elem.get_text(strip=True)
                                        if len(text) > 50:
                                            report['summary'] = text.replace('\n', ' ').strip()
                                            print(f">>> summary:{report['summary'][:50]}...")
                                            break
                                
                                # 提取完整內容 - 從「背景」標題之後
                                background_heading = content_container.find('h2', string=lambda text: text and '背景' in text)
                                if background_heading:
                                    full_content_texts = ["背景"]
                                    
                                    current = background_heading.next_sibling
                                    while current:
                                        if hasattr(current, 'get_text'):
                                            text = current.get_text(strip=True)
                                            if text and len(text) > 10:
                                                full_content_texts.append(text)
                                        current = current.next_sibling
                                    
                                    # 如果內容不夠，嘗試其他方法
                                    if len(full_content_texts) < 3:
                                        full_content_texts = ["背景"]
                                        parent = background_heading.parent
                                        found_bg = False
                                        for elem in parent.find_all(['h2', 'h3', 'h4', 'p']):
                                            if found_bg:
                                                text = elem.get_text(strip=True)
                                                if text and len(text) > 10:
                                                    full_content_texts.append(text)
                                            elif elem == background_heading:
                                                found_bg = True
                                    
                                    if len(full_content_texts) > 1:
                                        report['full_content'] = '\n\n'.join(full_content_texts)
                                        print(f">>> full_content:{report['full_content'][:50]}...")
                        except Exception as e:
                            print(f">>> 抓取文章內容時出錯: {e}")
                    
                    
                    # ⚡ 立即處理這篇文章：embedding -> 備份 -> ES上傳
                    total_processed += 1
                    
                    # 記錄第一篇（最新）文章標題用於更新歷史記錄
                    if first_article_title is None:
                        first_article_title = title
                    
                    result = embedding_and_save(report)
                    
                    # 統計結果
                    if result['status'] == 'success':
                        success_count += 1
                    elif result['status'] == 'skip':
                        skip_count += 1
                    else:
                        error_count += 1
                        
                    if result['backup']:
                        backup_count += 1
                    
                    # 簡短延遲避免過於頻繁的請求
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"處理第{page}頁第{i+1}篇文章時出錯: {e}")
                    continue
            
            if not duplicate_found:
                # 檢查是否有下一頁
                next_button = soup.select_one('a.next.page-numbers')
                if not next_button:
                    print("沒有找到下一頁按鈕，爬取完成")
                    break
                    
                page += 1
                time.sleep(1)  # 頁面間延遲
        
        # 更新爬蟲歷史記錄 - 記錄最新處理的文章標題
        if total_processed > 0 and first_article_title is not None and not duplicate_found:
            update_last_crawled_title(first_article_title)
            print(f">>> 已記錄最新處理文章標題：{first_article_title}")
        
        return {
            'total': total_processed,
            'success': success_count,
            'skip': skip_count,
            'error': error_count,
            'backup': backup_count
        }
        
    except Exception as e:
        print(f"\n❌ Pipeline失敗: {e}")
        raise

if __name__ == "__main__":
    # 執行pipeline
    print("啟動查核報告爬蟲")

    try:
        result = main_process(max_pages=40)
        
        if result:
            print(f"\n>>> 執行結果總結:")
            print(f"    處理文章: {result['total']} 篇")
            print(f"    上傳成功: {result['success']} 篇")
            print(f"    跳過重複: {result['skip']} 篇")
            print(f"    處理失敗: {result['error']} 篇")
            print(f"    備份成功: {result['backup']} 篇")
        
        print("\n✅ 查核報告更新完畢！")
        print("更新日期：",datetime.now().strftime("%Y-%m-%d %H:%M"))
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 查核報告更新失敗: {e}")
        import traceback
        traceback.print_exc()