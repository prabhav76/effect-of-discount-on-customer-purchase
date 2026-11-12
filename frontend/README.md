# Research dashboard

React + Vite frontend styled with Tailwind CSS and project-specific CSS. The dashboard data is generated from the project's processed analysis CSVs by `../analysis/export_dashboard_data.py`.

## Run locally

```bash
npm install
npm run dev
```

The `predev` and `prebuild` hooks refresh `public/data/dashboard.json` from the latest processed outputs. To refresh it manually, run:

```bash
npm run refresh-data
```

When raw transactions change, rerun the relevant analysis notebooks before refreshing the dashboard summary. `npm run build` creates a production bundle in `dist/`.
