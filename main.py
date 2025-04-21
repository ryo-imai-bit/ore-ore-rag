import duckdb
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "pfnet/plamo-embedding-1b", trust_remote_code=True
)
model = AutoModel.from_pretrained("pfnet/plamo-embedding-1b", trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# https://sora.shiguredo.jp/
docs = [
    "WebRTC による音声・映像・メッセージメッセージのリアルタイムな配信と、その録音・録画を実現します",
    "お客様ご自身のサーバーにインストールしてご利用いただくパッケージソフトウェアです",
    "株式会社時雨堂がフルスクラッチで開発しており、日本語によるサポートとドキュメントを提供します",
]


def main():
    # メモリ上にデータベースを作成
    print("duckdb.connect()")
    conn = duckdb.connect()
    conn.sql("INSTALL vss")
    conn.sql("LOAD vss")

    print("create table")
    conn.sql("CREATE SEQUENCE IF NOT EXISTS id_sequence START 1;")
    conn.sql(
        "CREATE TABLE IF NOT EXISTS sora_doc (id INTEGER DEFAULT nextval('id_sequence'), content TEXT, vector FLOAT[2048]);"
    )
    print("create table done")

    print("encode document")
    with torch.inference_mode():
        for doc, doc_embedding in zip(docs, model.encode_document(docs, tokenizer)):
            conn.execute(
                "INSERT INTO sora_doc (content, vector) VALUES (?, ?)",
                [
                    doc,
                    doc_embedding.cpu().squeeze().numpy().tolist(),
                ],
            )
    print("encode document done")

    query = "時雨堂について教えてください"
    print("query:", query)

    print("encode query")
    with torch.inference_mode():
        query_embedding = model.encode_query(query, tokenizer)
        print("query_embedding done")
        result = conn.sql(
            """
            SELECT content, array_cosine_distance(vector, ?::FLOAT[2048]) as distance
            FROM sora_doc
            ORDER BY distance
            """,
            params=[query_embedding.cpu().squeeze().numpy().tolist()],
        )
        print("result done")
        for row in result.fetchall():
            print("distance:", row[1], "|", row[0])


if __name__ == "__main__":
    main()
    # query: 時雨堂について教えてください
    # distance: 0.22656434774398804 | 株式会社時雨堂がフルスクラッチで開発しており、日本語によるサポートとドキュメントを提供します
    # distance: 0.39890891313552856 | お客様ご自身のサーバーにインストールしてご利用いただくパッケージソフトウェアです
    # distance: 0.5286199450492859 | WebRTC による音声・映像・メッセージメッセージのリアルタイムな配信と、その録音・録画を実現します
