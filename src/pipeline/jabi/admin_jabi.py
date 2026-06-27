from function.util.chroma_db import ChromaDB
from function.util.graph_db import GraphDB
from pipeline.services.embedding_util import EmbeddingModel
from config import *

chromadb = ChromaDB(collection_name=CHROMA_DB_NAME)
graphdb = GraphDB(NEO_URI, (NEO_USER, NEO_PASS))
embedding_model = EmbeddingModel()

if __name__=='__main__':
    ### 모든 데이터 출력 ###
    # print("--- ChromaDB (Vector DB) 데이터 ---")
    # all_items = chromadb.get_all_data()
    # print(f"총 {len(all_items)}개의 데이터가 있습니다.")
    # for item in all_items:
    #     print(item['document'], item['metadata'])
    
    # print("--- GraphDB (Neo4j) 데이터 ---")
    # all_graph_nodes = graphdb.get_all_nodes()
    # print("성공!")
    # print(f"GraphDB에 총 {len(all_graph_nodes)}개의 노드가 있습니다.")
    # for node_data in all_graph_nodes:
    #     # element_id, labels, 그리고 모든 속성(properties)이 출력됨
    #     print(f" - ID: {node_data['element_id']}")
    #     print(f"   Labels: {node_data['labels']}")
    #     print(f"   Properties: {node_data['properties']}")

    # ### 유사도 검색 테스트 ###
    # query = "엄마 생일 언제야?"
    # query_embedding = embedding_model.inference(context=query)
    # query_embedding = query_embedding.tolist()
    # similar_items = chromadb.query_similar(query_embedding)
    # print(f"총 {len(similar_items)}개의 유사한 데이터가 있습니다.")
    # for item in similar_items:
    #     print(item['document'], item['metadata'], item['distance'])

    # ### GraphDB ID 이용하여 노드 삭제 ###
    # event_id = "4:3b52b560-a9b5-47ab-a252-76baf116b41f:22"
    # graphdb.delete_event_by_id(event_id=event_id)

    # ### ChromaDB 컬렉션 삭제 ###
    # chromadb.delete_collection(collection_name=CHROMA_DB_NAME)

    ### GraphDB 삭제 ###
    graphdb.clear_database()
