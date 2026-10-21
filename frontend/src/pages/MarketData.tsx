import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { AlertTriangle, Database, Loader2, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";

type Dataset = {
  source: string;
  symbol: string;
  timeframe: string;
  dataset_type: string;
  start_date: string;
  end_date: string;
  row_count: number;
  first_timestamp: string | null;
  last_timestamp: string | null;
  updated_at: string;
  stored_bytes: number;
};

export function MarketData() {
  const { t, i18n } = useTranslation();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [storedBytes, setStoredBytes] = useState(0);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (initial = false) => {
    if (initial) setLoading(true);
    else setRefreshing(true);
    setError(null);
    try {
      const result = await api.getMarketDataDatasets();
      setDatasets(Array.isArray(result.datasets) ? result.datasets : []);
      setStoredBytes(Number(result.stored_bytes) || 0);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("marketData.loadError"));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [t]);

  useEffect(() => { void load(true); }, [load]);

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    const units = ["KB", "MB", "GB", "TB"];
    let value = bytes / 1024;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
    return `${new Intl.NumberFormat(i18n.language, { maximumFractionDigits: 1 }).format(value)} ${units[unit]}`;
  };
  const formatDate = (value: string | null) => {
    if (!value) return t("marketData.unknown");
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? value : new Intl.DateTimeFormat(i18n.language, { dateStyle: "medium", timeStyle: value.includes("T") ? "short" : undefined }).format(parsed);
  };

  return (
    <main className="min-h-screen p-6 lg:p-8">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
        <header className="flex flex-col gap-4 border-b pb-6 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <div className="mb-3 inline-flex items-center gap-2 rounded-md border px-2.5 py-1 text-xs font-medium text-muted-foreground">
              <Database className="h-3.5 w-3.5" />{t("marketData.badge")}
            </div>
            <h1 className="text-2xl font-semibold tracking-tight">{t("marketData.title")}</h1>
            <p className="mt-2 max-w-2xl text-sm text-muted-foreground">{t("marketData.subtitle")}</p>
          </div>
          <button type="button" onClick={() => void load()} disabled={refreshing} className="inline-flex items-center gap-2 self-start rounded-md border px-4 py-2 text-sm font-medium transition hover:bg-muted disabled:opacity-50 sm:self-auto">
            {refreshing ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}{t("marketData.refresh")}
          </button>
        </header>

        <section className="grid gap-3 sm:grid-cols-2">
          <Summary label={t("marketData.datasetCount")} value={loading ? "—" : String(datasets.length)} />
          <Summary label={t("marketData.storageUsed")} value={loading ? "—" : formatBytes(storedBytes)} />
        </section>

        {error && <section role="alert" className="rounded-md border border-warning/30 bg-warning/5 p-5">
          <div className="flex items-center gap-2 font-medium text-warning"><AlertTriangle className="h-5 w-5" />{t("marketData.unavailable")}</div>
          <p className="mt-2 text-sm text-muted-foreground">{error}</p>
          <button type="button" onClick={() => void load()} className="mt-3 rounded-md border px-3 py-1.5 text-sm hover:bg-muted">{t("marketData.retry")}</button>
        </section>}

        {loading && <div className="grid gap-3">{[1, 2, 3].map((item) => <div key={item} className="h-36 animate-pulse rounded-md border bg-muted/40" />)}</div>}
        {!loading && !error && datasets.length === 0 && <section className="rounded-md border border-dashed p-8 text-center">
          <Database className="mx-auto h-8 w-8 text-muted-foreground" />
          <h2 className="mt-3 font-medium">{t("marketData.emptyTitle")}</h2>
          <p className="mt-1 text-sm text-muted-foreground">{t("marketData.emptyBody")}</p>
        </section>}
        {!loading && !error && datasets.length > 0 && <section aria-label={t("marketData.title")} className="grid gap-3">
          {datasets.map((dataset, index) => <article key={`${dataset.source}:${dataset.symbol}:${dataset.timeframe}:${dataset.start_date}:${dataset.end_date}:${index}`} className="rounded-md border p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="font-mono text-base font-semibold">{dataset.symbol}</h2>
              <span className="rounded-full bg-primary/10 px-2.5 py-1 text-xs font-medium text-primary">{dataset.dataset_type === "perpetual_bundle" ? t("marketData.perpetualBundle") : t("marketData.ohlcv")}</span>
            </div>
            <dl className="mt-4 grid gap-x-6 gap-y-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
              <DataField label={t("marketData.source")} value={dataset.source} />
              <DataField label={t("marketData.timeframe")} value={dataset.timeframe} />
              <DataField label={t("marketData.coverage")} value={`${formatDate(dataset.first_timestamp || dataset.start_date)} — ${formatDate(dataset.last_timestamp || dataset.end_date)}`} />
              <DataField label={t("marketData.records")} value={new Intl.NumberFormat(i18n.language).format(dataset.row_count)} />
              <DataField label={t("marketData.size")} value={formatBytes(dataset.stored_bytes)} />
              <DataField label={t("marketData.updated")} value={formatDate(dataset.updated_at)} />
            </dl>
          </article>)}
        </section>}
      </div>
    </main>
  );
}

function Summary({ label, value }: { label: string; value: string }) {
  return <div className="rounded-md border p-4"><p className="text-sm text-muted-foreground">{label}</p><p className="mt-1 text-xl font-semibold">{value}</p></div>;
}

function DataField({ label, value }: { label: string; value: string }) {
  return <div className="min-w-0"><dt className="text-xs text-muted-foreground">{label}</dt><dd className="mt-1 break-words font-medium">{value}</dd></div>;
}
