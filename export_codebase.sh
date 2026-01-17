#!/usr/bin/env bash

# ============================
# CONFIG
# ============================
OUTPUT_FILE="codebase_dump.txt"

# ============================
# START
# ============================
echo "Generating ${OUTPUT_FILE} ..."

echo "Project root: $(pwd)" > "${OUTPUT_FILE}"
echo "Generated on: $(date)" >> "${OUTPUT_FILE}"

# ============================
# DIRECTORY STRUCTURE
# ============================
echo >> "${OUTPUT_FILE}"
echo "============================" >> "${OUTPUT_FILE}"
echo "DIRECTORY STRUCTURE" >> "${OUTPUT_FILE}"
echo "============================" >> "${OUTPUT_FILE}"

tree -a \
  -I 'data|datasets|.git|.venv|venv|env|node_modules|__pycache__' \
  >> "${OUTPUT_FILE}"

# ============================
# FILE CONTENTS
# ============================
echo >> "${OUTPUT_FILE}"
echo "============================" >> "${OUTPUT_FILE}"
echo "SOURCE FILE CONTENTS" >> "${OUTPUT_FILE}"
echo "============================" >> "${OUTPUT_FILE}"

find . \
  \( -path ./data \
     -o -path ./datasets \
     -o -path ./.git \
     -o -path ./.venv \
     -o -path ./venv \
     -o -path ./env \
     -o -path ./node_modules \
     -o -path ./__pycache__ \
     -o -path "./${OUTPUT_FILE}" \
     -o -path "./solr/*" ! -name "managed-schema" \
  \) -prune -o \
  -type f \( \
    -name "*.sh" \
    -o -name "*.py" \
    -o -name "*.yaml" \
    -o -name "*.yml" \
    -o -name "*.json" \
    -o -name "*.toml" \
    -o -name "*.ini" \
    -o -name "*.cfg" \
    -o -name "*.md" \
    -o -name "*.txt" \
    -o -name "managed-schema" \
  \) -print \
| sort \
| while read -r file; do
    echo >> "${OUTPUT_FILE}"
    echo "----------------------------------------" >> "${OUTPUT_FILE}"
    echo "FILE: ${file}" >> "${OUTPUT_FILE}"
    echo "----------------------------------------" >> "${OUTPUT_FILE}"
    cat "${file}" >> "${OUTPUT_FILE}"
  done

echo >> "${OUTPUT_FILE}"
echo "DONE." >> "${OUTPUT_FILE}"

echo "Output written to ${OUTPUT_FILE}"
