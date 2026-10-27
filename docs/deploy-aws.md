# Deploying to AWS (EC2 + Docker Compose)

One EC2 virtual machine runs the three containers from `docker-compose.yml`:

```
Internet ──► :443 proxy (Caddy, automatic HTTPS) ──► frontend (Streamlit) ──► backend (FastAPI)
                    └── only /api/documents/<id>/file (PDF links) ──► backend
```

Only the proxy is reachable from outside. The backend API, including `/docs`, is private:
the frontend calls it inside Docker's network. Plan on about an hour the first time.

> **Status:** this setup is written to match the tested local application, but at the time
> of writing it had not yet been run on AWS. Step 9 validates the proxy configuration
> before starting; report anything that fails.

## Before you start

- **Login configured** (README, "Login"): an admin account in `data/users.json` and an
  `AUTH_SECRET_KEY`. Never deploy with `AUTH_ENABLED=false`.
- **A domain name for HTTPS.** Let's Encrypt certificates need a domain pointing at the
  server. Use one you own, or a free dynamic-DNS subdomain (for example DuckDNS). Without
  a domain the app only works over plain HTTP, which sends passwords unencrypted; don't
  use it that way except to test.
- **Costs.** A `t3.medium` running all month costs roughly $30-35 on demand in US
  regions, plus storage and a small hourly charge for the public IPv4 address. Check the
  [AWS Pricing Calculator](https://calculator.aws/) for your region, and set an
  **AWS Budget** alert (for example $10) before launching anything.
- **Free-tier LLM.** The Gemini free tier allows about 5 requests per minute, and Google
  may use free-tier content to improve its products. Share the demo with a few people,
  and only with documents you are allowed to send to Google.

## 1. Launch the server

AWS console → **EC2** → **Launch instance**:

| Setting | Value |
|---|---|
| Name | `hybrid-rag` |
| Image | **Ubuntu Server 24.04 LTS** (64-bit x86) |
| Instance type | **t3.medium** (2 vCPU, 4 GB RAM). Smaller instances run out of memory loading PyTorch and the models |
| Key pair | Create one, download the `.pem` file, keep it safe |
| Storage | **30 GB gp3** |
| Security group | SSH (22) from **My IP** only; HTTP (80) and HTTPS (443) from **Anywhere** (Let's Encrypt must reach port 80 or 443 to issue the certificate) |

Then **Elastic IPs → Allocate → Associate** it with the instance, so the address survives
restarts.

## 2. Point your domain at it

At your DNS provider, create an **A record** for your (sub)domain with the Elastic IP as
its value. Check it from PowerShell (it may take a few minutes):

```powershell
Resolve-DnsName rag.example.com
```

## 3. Connect

From PowerShell on your PC (Windows includes OpenSSH). Restrict the key file first, or
SSH refuses to use it:

```powershell
icacls .\hybrid-rag.pem /inheritance:r /grant:r "$($env:USERNAME):(R)"
ssh -i .\hybrid-rag.pem ubuntu@<elastic-ip>
```

All commands below run **on the server** unless marked otherwise.

## 4. Prepare the server

```bash
sudo apt-get update && sudo apt-get upgrade -y

# 2 GB of swap: a safety margin for building the image and loading models
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Install Docker from Docker's official repository
([docs](https://docs.docker.com/engine/install/ubuntu/)):

```bash
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker ubuntu
exit
```

Reconnect with `ssh` so the group change applies, then check `docker compose version`.

## 5. Get the code

```bash
git clone https://github.com/SubhrojyotiSarkar/hybrid-search-rag-citation-verification.git
cd hybrid-search-rag-citation-verification
```

## 6. Copy your data and settings (on your PC)

From the **project folder on your PC**, in PowerShell. This copies your documents,
indexes and accounts (`data/`), your evaluation folder, and `.env`:

```powershell
scp -i .\path\to\hybrid-rag.pem -r data evaluation .env ubuntu@<elastic-ip>:~/hybrid-search-rag-citation-verification/
```

(Alternatively, start with an empty `data/` and upload documents through the app.)

## 7. Edit the settings (on the server)

```bash
chmod 600 .env
nano .env
```

Check or set:

```
AUTH_ENABLED=true
AUTH_SECRET_KEY=<a long secret>   # keep yours or generate a new one (logs everyone out)
SITE_ADDRESS=rag.example.com      # your domain: enables automatic HTTPS
GEMINI_API_KEY=<your key>
RETRIEVAL_MODE=hybrid_rerank      # optional
MAX_RECOVERY_ATTEMPTS=0           # recommended on the free tier
```

`DATA_DIR`, `EVALUATION_DIR`, `BACKEND_URL` and `PUBLIC_BACKEND_URL` are set by
`docker-compose.yml`; their values in `.env` do not matter on the server.

## 8. Accounts

If you copied `data/users.json`, your accounts are already there. Otherwise, or to add a
read-only account for someone else:

```bash
docker compose run --rm backend python -m backend.auth_admin add-user <name> --role admin
docker compose run --rm backend python -m backend.auth_admin add-user recruiter --role viewer
```

(The first `docker compose run` builds the image, which takes several minutes.)

## 9. Validate, then start

```bash
docker compose run --rm proxy caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
docker compose up -d --build
docker compose ps                 # wait until backend and frontend show "healthy"
docker compose logs -f backend    # Ctrl+C to stop following
```

The first start downloads the models, which takes a few minutes. The proxy waits until
the frontend is healthy, then obtains the certificate (`docker compose logs proxy`).

## 10. Check it works

Open `https://rag.example.com` and run the main checks from the final test plan:

- [ ] The browser shows a valid padlock (HTTPS) and the login form
- [ ] Admin login works; a viewer account is read-only
- [ ] Ask a question: verified answer with citations; **open PDF** opens at the right page
- [ ] `https://rag.example.com/docs` and `https://rag.example.com/api/documents` do **not**
      show the API (the backend is private)
- [ ] Five wrong passwords lock the account

## Running it

| Task | Command (on the server, in the project folder) |
|---|---|
| Logs | `docker compose logs -f backend` (every line carries the request ID shown in UI errors) |
| Update to the latest code | `git pull && docker compose up -d --build` |
| Restart | `docker compose restart` |
| Index consistency | `docker compose run --rm backend python -m backend.maintenance status` (stop the backend first for `rebuild-index` / `reprocess`) |
| Back up data | `tar czf backup-$(date +%F).tgz data evaluation`, then from your PC: `scp -i key.pem ubuntu@<ip>:~/hybrid-search-rag-citation-verification/backup-*.tgz .` (or EBS snapshots in the EC2 console) |
| OS security updates | Ubuntu installs them automatically (`unattended-upgrades`); reboot occasionally: `sudo reboot` |

**Save money:** stop the instance in the EC2 console when you are not demoing (you then
pay only for storage and the Elastic IP). Containers restart automatically when the
instance starts again (`restart: unless-stopped`).

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| Browser can't connect | Security group or DNS | Check ports 80/443 are open; `Resolve-DnsName` returns the Elastic IP |
| Certificate error / proxy log shows ACME errors | DNS not pointing here yet, or port 80/443 blocked | Fix DNS or the security group, then `docker compose restart proxy` |
| 502 Bad Gateway | Backend or frontend still starting | Wait; `docker compose ps` should show `healthy` |
| Containers restart in a loop, `Killed` in logs | Out of memory | Use t3.medium or larger; keep the swap file |
| Backend exits with "AUTH_ENABLED=true but …" | Login misconfigured (by design it refuses to start) | Set `AUTH_SECRET_KEY` / create an admin (step 8) |
| "rate limit or free-tier quota" in answers | Gemini free tier | Wait a minute; set `MAX_RECOVERY_ATTEMPTS=0` |

## Removing everything

To stop all charges: EC2 console → **terminate** the instance, **release** the Elastic IP,
and delete any leftover volumes and snapshots. Download a backup first if you want to
keep your data.
