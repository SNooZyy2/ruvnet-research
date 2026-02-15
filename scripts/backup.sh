#!/usr/bin/env bash
# Research database backup script
# Usage: ./scripts/backup.sh [--compress] [--max-backups N]

set -euo pipefail

DB_PATH="/home/snoozyy/ruvnet-research/db/research.db"
BACKUP_DIR="/home/snoozyy/ruvnet-research/db/backups"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
MAX_BACKUPS=10
COMPRESS=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --compress) COMPRESS=true; shift ;;
    --max-backups) MAX_BACKUPS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$BACKUP_DIR"

BACKUP_FILE="$BACKUP_DIR/research-${TIMESTAMP}.db"

# Use better-sqlite3's backup API (safe even during active writes)
node -e "
const Database = require('better-sqlite3');
const db = new Database('${DB_PATH}');
db.backup('${BACKUP_FILE}').then(() => {
  console.log('SQLite backup API completed');
  db.close();
}).catch(err => {
  console.error('Backup failed:', err.message);
  db.close();
  process.exit(1);
});
"

if $COMPRESS; then
  gzip "$BACKUP_FILE"
  BACKUP_FILE="${BACKUP_FILE}.gz"
fi

# Rotate old backups, keeping only the most recent N
ls -1t "$BACKUP_DIR"/research-*.db* 2>/dev/null | tail -n +$((MAX_BACKUPS + 1)) | xargs -r rm --

SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "Backup complete: $BACKUP_FILE ($SIZE)"
echo "Backups retained: $(ls -1 "$BACKUP_DIR"/research-*.db* 2>/dev/null | wc -l)/$MAX_BACKUPS"
