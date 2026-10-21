import { render, screen } from "@testing-library/react";
import { MarketData } from "../MarketData";

const apiMock = vi.hoisted(() => ({ getMarketDataDatasets: vi.fn() }));

vi.mock("@/lib/api", () => ({ api: apiMock }));

describe("MarketData page", () => {
  beforeEach(() => apiMock.getMarketDataDatasets.mockReset());

  it("shows saved datasets and total stored data", async () => {
    apiMock.getMarketDataDatasets.mockResolvedValue({
      dataset_count: 1,
      stored_bytes: 2048,
      datasets: [{
        source: "okx", symbol: "ETH-USDT-SWAP", timeframe: "1H",
        dataset_type: "perpetual_bundle", start_date: "2024-01-01", end_date: "2024-01-02",
        row_count: 24, first_timestamp: "2024-01-01T00:00:00Z",
        last_timestamp: "2024-01-02T00:00:00Z", updated_at: "2024-01-03T00:00:00Z",
        stored_bytes: 2048,
      }],
    });

    render(<MarketData />);

    expect(await screen.findByText("ETH-USDT-SWAP")).toBeInTheDocument();
    expect(screen.getByText("okx")).toBeInTheDocument();
    expect(screen.getByText("Perpetual data bundle")).toBeInTheDocument();
    expect(screen.getByText("24")).toBeInTheDocument();
    expect(screen.getAllByText("2 KB")).toHaveLength(2);
  });

  it("explains an empty inventory", async () => {
    apiMock.getMarketDataDatasets.mockResolvedValue({ datasets: [], dataset_count: 0, stored_bytes: 0 });
    render(<MarketData />);
    expect(await screen.findByText("No market data saved yet")).toBeInTheDocument();
  });
});
