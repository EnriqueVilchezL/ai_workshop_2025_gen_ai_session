# Global configuration settings.
model = "llama3.1"
embedding_model = "llama3.1"
model_provider = "ollama"
user = "France Du Pont"

db_dir = "db"
sources_dir = "dummy_sources"

db_search_kwargs={"k": 3, "score_threshold": 0.03}
db_search_type = "similarity_score_threshold"