# Deploy to Azure Container Apps (from GitHub)

The web app ships as a **serve-only** container (see the table in the root `README.md`): it serves
the baked-in ~9 MB synthetic world and the live recommenders, reads your external Postgres for the
Explore pages, and makes **no** build or database writes on startup. Secrets are passed as
Container App environment variables — never baked into the image or committed.

## Prerequisites

- `az login` (done) and a subscription selected: `az account show`
- The Azure Container Apps extension: `az extension add --name containerapp --upgrade`
- Providers registered (one-time):
  `az provider register -n Microsoft.App && az provider register -n Microsoft.OperationalInsights`

## One command (build from this repo, GitHub-driven CI)

Run from `webapp/`. This creates the resource group, an Azure Container Registry, a Container Apps
environment, builds the image **from the GitHub repo**, deploys it, and wires a GitHub Actions
workflow so future pushes redeploy automatically (the NewsRAG pattern):

```bash
az containerapp up \
  --name food-recs-lab \
  --resource-group rg-food-recs \
  --location westeurope \
  --environment env-food-recs \
  --repo hallaai/rest-recommender \
  --context-path webapp \
  --ingress external \
  --target-port 8000 \
  --query properties.configuration.ingress.fqdn -o tsv
```

`--context-path webapp` points the build at the folder that holds the `Dockerfile`. The command
prints the public HTTPS hostname (`https://food-recs-lab.<hash>.westeurope.azurecontainerapps.io`).

> Alternative (build locally, no data on GitHub): `az acr build` the image from `webapp/` to your
> ACR, then `az containerapp up --image <acr>.azurecr.io/food-recs-lab:latest`. Use this only if you
> decide not to commit the 9 MB `data/`.

## Set the secrets (required)

The app withholds everything until login, and Explore needs Postgres, so set these after the first
deploy (values come from `webapp/.env` — do not commit them):

```bash
az containerapp secret set -n food-recs-lab -g rg-food-recs --secrets \
  database-url="postgresql://USER:PASS@HOST:PORT/DB" \
  app-password="..." \
  gemini-api-key="..."

az containerapp update -n food-recs-lab -g rg-food-recs \
  --set-env-vars \
    DATABASE_URL=secretref:database-url \
    APP_USERNAME=admin \
    APP_PASSWORD=secretref:app-password \
    GEMINI_API_KEY=secretref:gemini-api-key
```

`GEMINI_API_KEY` is optional for serving (only the LLM-rerank experiment uses it); omit it to run
with no Gemini spend. `DATABASE_URL` is needed for the Explore pages; without it those pages show
"database unavailable" but Recommend/Compare still work.

## Health & scaling

- Liveness probe: `GET /healthz` (unauthenticated, returns `{"status":"ok"}`, no DB touch).
- The app holds fitted recommenders in memory and warms the LightGBM pipeline in the background,
  so keep **min replicas = 1** to avoid cold re-fits on every request:
  `az containerapp update -n food-recs-lab -g rg-food-recs --min-replicas 1 --max-replicas 2`

## Update

With CI wired by `az containerapp up --repo`, push to the branch and GitHub Actions rebuilds and
redeploys. Or redeploy on demand: `az containerapp up ...` again with the same names.
