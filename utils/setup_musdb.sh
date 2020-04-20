ARCHIVE_NAME="musdb18.zip"
MUSDB_DIR="musdb18"

ROOT_DIR="$( cd "$(dirname "$0")/.." >/dev/null 2>&1 ; pwd -P )"
DATASETS_DIR="$ROOT_DIR/datasets"

ARCHIVE_PATH="$DATASETS_DIR/$ARCHIVE_NAME"
MUSDB_PATH="$DATASETS_DIR/$MUSDB_DIR"

mkdir -p $MUSDB_PATH

unzip -u $ARCHIVE_PATH -d $MUSDB_PATH && rm $ARCHIVE_PATH
echo "Extracted $MUSDB_DIR dataset in $MUSDB_PATH"

echo "Computing wav files..."
musdbconvert $MUSDB_PATH $MUSDB_PATH
