from confluent_kafka.admin import AdminClient, NewTopic

conf = {'bootstrap.servers': 'localhost:9092'}
admin_client = AdminClient(conf)

new_topic = [NewTopic("chat-topic", num_partitions=1, replication_factor=1), NewTopic("log-topic", num_partitions=1, replication_factor=1)]

fs = admin_client.create_topics(new_topic)

for topic, f in fs.items():
    try:
        f.result()  # 성공하면 None 반환
        print(f"토픽 '{topic}' 생성 완료!")
    except Exception as e:
        print(f"토픽 생성 실패: {e}")
