from elasticsearch import Elasticsearch
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import os
from datetime import datetime



### 上傳「單一document」到Elasticsearch並以pid偵測是否重複，如果有重複就直接覆寫，沒有重複就創建新的
class ESUploader:
    def __init__(self):
        """Initialize Elasticsearch connection using environment variables"""
        load_dotenv()
        self.es_username = os.getenv('es_username')
        self.es_password = os.getenv('es_password')
        self.es_client = Elasticsearch(
            "https://media-vector.es.asia-east1.gcp.elastic-cloud.com",  
            basic_auth=(self.es_username, self.es_password)
        )

    def _format_date(self, date_str: str) -> str:
        """將日期字串轉換為 Elasticsearch 可接受的格式"""
        if not date_str:
            return None
        try:
            # 嘗試解析不同的日期格式
            date_formats = [
                "%Y/%m/%d %H:%M",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y/%m/%d",
                "%Y-%m-%d"
            ]
            
            for fmt in date_formats:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    return date_obj.strftime("%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    continue
            
            print(f"[WARN] 無法解析日期格式: {date_str}")
            return None
        except Exception as e:
            print(f"[ERROR] 日期格式轉換錯誤: {str(e)}")
            return None

    def get_index_fields(self, index: str) -> List[str]:
        """獲取索引的所有欄位名稱"""
        try:
            mapping = self.es_client.indices.get_mapping(index=index)
            properties = mapping[index]['mappings'].get('properties', {})
            return list(properties.keys())
        except Exception as e:
            return []

    def upload_single_document(self, document: Dict[str, Any], index: str, strict_fields: bool = False) -> Dict[str, Any]:
        """
        Upload a single document to Elasticsearch
        
        Args:
            document (Dict[str, Any]): The document to upload
            index (str): The target Elasticsearch index
            strict_fields (bool): If True, only upload fields that exist in the index
            
        Returns:
            Dict[str, Any]: Standard response format with ResultData, Result and Message
        """
        try:
            print(f"[INFO] 開始處理文件上傳，目標索引: {index}")
            
            # 檢查文件是否為空
            if not document:
                print("[ERROR] 文件為空")
                return {
                    "ResultData": None,
                    "Result": "N",
                    "Message": "Document cannot be empty"
                }

            # 檢查文件是否包含 pid
            if 'pid' not in document:
                print("[ERROR] 文件缺少 pid 欄位")
                return {
                    "ResultData": None,
                    "Result": "N",
                    "Message": "Document must contain 'pid' field"
                }

            print(f"[INFO] 開始清理文件，處理 None 值")
            print(f"[DEBUG] 原始文件大小: {len(document)} 欄位")
            
            # 定義要排除的時間相關欄位
            time_fields_to_exclude = {
                'time_articut_tagger',
                'time_date_conversion',
                'time_prewrite_summary',
                'time_text_embeddings',
                'total_processing_time'
            }
            
            # Clean the document to ensure no None values are present
            cleaned_document = {}
            for key, value in document.items():
                # 跳過時間相關欄位
                if key in time_fields_to_exclude:
                    print(f"[DEBUG] 跳過時間相關欄位: {key}")
                    continue
                    
                try:
                    if value is None:
                        print(f"[DEBUG] 發現 None 值欄位: {key}")
                        cleaned_document[key] = ""  # Convert None to empty string
                    elif key in ['dt', 'date_created'] and isinstance(value, str):
                        # 處理日期欄位
                        formatted_date = self._format_date(value)
                        if formatted_date:
                            cleaned_document[key] = formatted_date
                        else:
                            cleaned_document[key] = value
                    elif isinstance(value, (list, tuple)):
                        print(f"[DEBUG] 處理列表/元組欄位: {key}, 長度: {len(value)}")
                        # 處理列表或元組中的 None 值
                        cleaned_list = []
                        for item in value:
                            if item is None:
                                print(f"[DEBUG] 在列表欄位 {key} 中發現 None 值")
                                cleaned_list.append('')
                            else:
                                cleaned_list.append(item)
                        cleaned_document[key] = cleaned_list
                    elif isinstance(value, dict):
                        print(f"[DEBUG] 處理字典欄位: {key}, 鍵的數量: {len(value)}")
                        # 處理字典中的 None 值
                        cleaned_dict = {}
                        for k, v in value.items():
                            if v is None:
                                print(f"[DEBUG] 在字典欄位 {key} 中發現 None 值，鍵: {k}")
                                cleaned_dict[k] = ''
                            else:
                                cleaned_dict[k] = v
                        cleaned_document[key] = cleaned_dict
                    else:
                        cleaned_document[key] = value
                except Exception as e:
                    print(f"[ERROR] 處理欄位 {key} 時發生錯誤: {str(e)}")
                    print(f"[ERROR] 欄位 {key} 的值類型: {type(value)}")
                    raise e

            print(f"[INFO] 文件清理完成，清理後大小: {len(cleaned_document)} 欄位")

            # 檢查索引是否存在，如果不存在則創建
            if not self.es_client.indices.exists(index=index):
                # 創建索引時設定 mapping，確保 pid 欄位存在
                mapping = {
                    "mappings": {
                        "properties": {
                            "pid": {"type": "text"},
                            "category": {"type": "keyword"},  # 使用 keyword 類型以便精確匹配
                            "entity": {
                                "type": "object"
                            },
                            "articut_result_obj": {
                                "type": "object",
                                "properties": {
                                    "pos": {"type": "keyword"},
                                    "text": {"type": "text"},
                                    "tag": {"type": "keyword"},
                                    "idx": {"type": "integer"}
                                }
                            }
                        }
                    }
                }
                self.es_client.indices.create(index=index, body=mapping)

            # 如果啟用了嚴格欄位檢查
            filtered_document = cleaned_document.copy()
            ignored_fields = []
            
            if strict_fields:
                existing_fields = self.get_index_fields(index)
                # 過濾掉不存在於索引中的欄位
                filtered_document = {
                    k: v for k, v in cleaned_document.items() 
                    if k in existing_fields
                }
                # 記錄被忽略的欄位
                ignored_fields = [
                    k for k in cleaned_document.keys() 
                    if k not in existing_fields
                ]

            # 檢查是否已存在相同 pid 的文件
            query = {
                "query": {
                    "term": {
                        "pid": filtered_document['pid']
                    }
                }
            }
            
            search_result = self.es_client.search(index=index, body=query)
            
            # 如果找到現有文件，使用其 _id 更新文件
            if search_result['hits']['total']['value'] > 0:
                existing_doc_id = search_result['hits']['hits'][0]['_id']
                response = self.es_client.index(
                    index=index,
                    id=existing_doc_id,
                    document=filtered_document
                )
                operation_type = "updated"
            else:
                # 如果沒有找到現有文件，創建新文件
                response = self.es_client.index(
                    index=index,
                    document=filtered_document
                )
                operation_type = "created"

            # 強制刷新索引以確保文件立即可見
            self.es_client.indices.refresh(index=index)

            result_data = {
                "response": response,
                "operation_type": operation_type,
                "document_id": response["_id"],
                "index": response["_index"],
                "version": response["_version"]
            }

            # 如果有被忽略的欄位，加入到結果中
            if ignored_fields:
                result_data["ignored_fields"] = ignored_fields

            return {
                "ResultData": result_data,
                "Result": "Y",
                "Message": (f"Document successfully {operation_type}" +
                          (f" (ignored fields: {', '.join(ignored_fields)})" if ignored_fields else ""))
            }

        except Exception as e:
            return {
                "ResultData": None,
                "Result": "N",
                "Message": f"Failed to upload document: {str(e)}"
            }

if __name__ == "__main__":
    # 創建上傳器實例
    uploader = ESUploader()
    
    # 測試文件 (加入一些可能不存在的欄位)
    test_document = {
        "pid": "doc123",
        "title": "測試文件",
        "content": "這是一個測試文件的內容",
        "timestamp": "2024-03-20",
        "tags": ["test", "demo"],
        "author": {
            "name": "測試作者",
            "email": "test@example.com"
        },
        "non_existing_field": "這個欄位可能不存在於索引中"
    }
    
    # 指定要上傳的索引
    test_index = "test_documents"
    
    # 使用嚴格欄位檢查上傳文件
    result = uploader.upload_single_document(
        document=test_document,
        index=test_index,
        strict_fields=True  # 啟用嚴格欄位檢查
    )
    
    # 輸出結果
    if result["Result"] == "Y":
        print("文件操作成功！")
        print(f"操作類型: {result['ResultData']['operation_type']}")
        print(f"文件ID: {result['ResultData']['document_id']}")
        print(f"索引名稱: {result['ResultData']['index']}")
        print(f"版本: {result['ResultData']['version']}")
        if "ignored_fields" in result["ResultData"]:
            print(f"被忽略的欄位: {', '.join(result['ResultData']['ignored_fields'])}")
        print(f"訊息: {result['Message']}")
    else:
        print(f"上傳失敗：{result['Message']}")