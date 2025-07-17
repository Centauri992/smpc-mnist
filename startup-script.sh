#!/bin/bash
set -e

# 1. Update and install dependencies
apt-get update
apt-get install -y python3-pip git python3-venv

# 2. Clone repo
REPO="smpc-mnist"

cd /root
git clone https://github.com/Centauri992/$REPO.git
cd /root/$REPO

# 3. Set up virtualenv
python3 -m venv smpc-venv
source smpc-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Fetch instance attributes
PARTY_ID="$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/attributes/party_id -H 'Metadata-Flavor: Google')"
BUCKET="$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket -H 'Metadata-Flavor: Google')"
TOTAL="$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/attributes/total -H 'Metadata-Flavor: Google')"
LOCAL_IP="$(hostname -I | awk '{print $1}')"

# 5. Register IP for coordination (one file per party)
echo "$LOCAL_IP" | gsutil cp - gs://$BUCKET/party-$PARTY_ID

# 6. Wait for all 3 IPs to appear
while [ $(gsutil ls gs://$BUCKET/party-* | wc -l) -lt $TOTAL ]; do
  sleep 2
done

# 7. Read all party IPs, build MPyC -P arguments
HOSTS=""
for i in $(seq 0 $(($TOTAL - 1))); do
  IP=$(gsutil cat gs://$BUCKET/party-$i)
  HOSTS="${HOSTS}${IP},"
done
HOSTS=${HOSTS::-1}

PORT=12345  # Use any available port; must be same for all
PARTY_ADDRS=""
for ip in $(echo $HOSTS | tr ',' ' '); do
  PARTY_ADDRS="$PARTY_ADDRS -P $ip:$PORT"
done

echo "PARTY $PARTY_ID running on $LOCAL_IP with party addresses: $PARTY_ADDRS"

BATCH=10
TOTAL_IMAGES=10000

cd /root/$REPO

source smpc-venv/bin/activate

# 8. Loop over all protocols
for PROTO in 0 1 2; do
  OUT="results_party${PARTY_ID}_k${PROTO}.txt"
  echo "Protocol d_k_star = $PROTO" > $OUT
  for OFFSET in $(seq 0 $BATCH $((TOTAL_IMAGES-1))); do
    python3 smpc.py -M $TOTAL -I $PARTY_ID $PARTY_ADDRS -b $BATCH -o $OFFSET -d $PROTO >> $OUT
  done
  gsutil cp $OUT gs://$BUCKET/
done

# 9. Toft's protocol
OUT="results_party${PARTY_ID}_toft.txt"
echo "Protocol Toft (built-in MPyC)" > $OUT
for OFFSET in $(seq 0 $BATCH $((TOTAL_IMAGES-1))); do
  python3 smpc.py -M $TOTAL -I $PARTY_ID $PARTY_ADDRS -b $BATCH -o $OFFSET --no-legendre >> $OUT
done
gsutil cp $OUT gs://$BUCKET/

# 10. Write done marker
gsutil cp <(echo "done") gs://$BUCKET/party-${PARTY_ID}-done

# 11. Wait for all to finish, then shutdown
while [ $(gsutil ls gs://$BUCKET/party-*-done | wc -l) -lt $TOTAL ]; do
  sleep 5
done

INSTANCE_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
ZONE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F/ '{print $NF}')
gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE --quiet

