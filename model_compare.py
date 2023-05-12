import train_model_test

not_running = {}
# "albert_kor_base", "klue_bert_base", "klue_roberta_large", "snunlp_kr_electra" done
# ["gogamza/kobart-base-v2", "skt_kobert_v1", "beomi_kcbert", "skt_ko_gpt"] errors
models = ["beomi_kcelectra_base_for_test", "kykim_electra_kor_base_for_test", "monologg_koelectera_base_for_test", "xlm_roberta_large_for_test", "skt_kobert_v1_for_test"]

for model in models:
    try:
        print(f"---------------------- {model} start ----------------------")
        train_model_test.main(model_name=model)
        print(f"---------------------- {model} finish ----------------------")
    except Exception as e:
        not_running[model] = e
        print(f"---------------------- {model} stopped as {e}----------------------")


for i, j in not_running.items():
    print(f"{i} failed with an error - {j}")
