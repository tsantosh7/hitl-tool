CREATE TABLE IF NOT EXISTS model_predictions (
  document_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  model TEXT,
  k_shot INTEGER,
  seed INTEGER,
  prediction_output_1 JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (document_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_model_predictions_run_id
  ON model_predictions(run_id);


CREATE TABLE IF NOT EXISTS topic_runs (
  run_id UUID PRIMARY KEY,
  project_id UUID NULL REFERENCES projects(project_id),
  name TEXT NOT NULL,
  topic_schema_version TEXT NOT NULL DEFAULT 'topics-v1',
  method TEXT NOT NULL DEFAULT 'external',
  model TEXT NULL,
  params JSONB NOT NULL DEFAULT '{}'::jsonb,
  is_active BOOLEAN NOT NULL DEFAULT FALSE,
  created_by TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_topic_runs_project_id ON topic_runs(project_id);
CREATE INDEX IF NOT EXISTS idx_topic_runs_is_active ON topic_runs(is_active);

CREATE TABLE IF NOT EXISTS document_topics (
  run_id UUID NOT NULL REFERENCES topic_runs(run_id) ON DELETE CASCADE,
  document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
  topic_key TEXT NOT NULL,
  topic_label TEXT NOT NULL,
  score DOUBLE PRECISION NULL,
  source TEXT NOT NULL DEFAULT 'model',
  evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (run_id, document_id, topic_key)
);

CREATE INDEX IF NOT EXISTS idx_document_topics_run_id ON document_topics(run_id);
CREATE INDEX IF NOT EXISTS idx_document_topics_document_id ON document_topics(document_id);
CREATE INDEX IF NOT EXISTS idx_document_topics_label ON document_topics(topic_label);


CREATE TABLE IF NOT EXISTS doc_embeddings (
  document_id TEXT PRIMARY KEY REFERENCES documents(document_id) ON DELETE CASCADE,
  embedding_dim INT NOT NULL,
  model TEXT NOT NULL,
  embedding BYTEA NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_doc_embeddings_model ON doc_embeddings(model);


ALTER TABLE document_topics
  ADD COLUMN IF NOT EXISTS assignment_type TEXT NOT NULL DEFAULT 'auto',   -- human|auto
  ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active',          -- active|deleted|rejected
  ADD COLUMN IF NOT EXISTS created_by TEXT NULL,
  ADD COLUMN IF NOT EXISTS updated_by TEXT NULL,
  ADD COLUMN IF NOT EXISTS reason TEXT NULL;

-- Recommended index for propagation queries:
CREATE INDEX IF NOT EXISTS idx_document_topics_run_topic_type_status
  ON document_topics(run_id, topic_key, assignment_type, status);

CREATE INDEX IF NOT EXISTS idx_document_topics_doc
  ON document_topics(run_id, document_id);
