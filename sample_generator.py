import io
import pandas as pd


def generate_sample_excel_bytes() -> bytes:
    """
    Generate a comprehensive multi-sheet Financial Report (.xlsx) for 'Apex Cloud Technologies Inc. - FY2026'
    containing an Income Statement, Balance Sheet, Cash Flow Statement, and Management Commentary.
    """
    output = io.BytesIO()

    income_df = pd.DataFrame(
        {
            "Line Item ($ in Millions)": [
                "Total Revenue",
                "Cost of Goods Sold",
                "Gross Profit",
                "Research & Development (R&D)",
                "Sales & Marketing (S&M)",
                "General & Administrative (G&A)",
                "Operating Income",
                "Interest & Other Expense",
                "Pre-Tax Income",
                "Income Tax Provision",
                "Net Income",
            ],
            "FY2025 ($M)": [980.0, 294.0, 686.0, 175.0, 210.0, 81.0, 220.0, -12.0, 208.0, -41.6, 166.4],
            "FY2026 ($M)": [1250.0, 325.0, 925.0, 215.0, 245.0, 90.0, 375.0, -10.0, 365.0, -73.0, 292.0],
            "YoY Change (%)": [
                "+27.6%",
                "+10.5%",
                "+34.8%",
                "+22.9%",
                "+16.7%",
                "+11.1%",
                "+70.5%",
                "-16.7%",
                "+75.5%",
                "+75.5%",
                "+75.5%",
            ],
        }
    )

    balance_df = pd.DataFrame(
        {
            "Balance Sheet Item ($ in Millions)": [
                "Cash and Cash Equivalents",
                "Accounts Receivable",
                "Prepaid Expenses & Other",
                "Total Current Assets",
                "Property, Plant & Equipment (Net)",
                "Intangible Assets & Goodwill",
                "Total Assets",
                "Accounts Payable",
                "Accrued Expenses & Deferred Revenue",
                "Total Current Liabilities",
                "Long-Term Debt",
                "Total Liabilities",
                "Total Shareholders Equity",
            ],
            "FY2025 ($M)": [410.0, 185.0, 45.0, 640.0, 210.0, 350.0, 1200.0, 95.0, 225.0, 320.0, 280.0, 600.0, 600.0],
            "FY2026 ($M)": [620.0, 230.0, 50.0, 900.0, 240.0, 360.0, 1500.0, 110.0, 250.0, 360.0, 240.0, 600.0, 900.0],
        }
    )

    cashflow_df = pd.DataFrame(
        {
            "Cash Flow Statement ($ in Millions)": [
                "Cash from Operating Activities",
                "Capital Expenditures (CapEx)",
                "Free Cash Flow",
                "Debt Repayment",
                "Net Change in Cash",
            ],
            "FY2025 ($M)": [245.0, -55.0, 190.0, -20.0, 170.0],
            "FY2026 ($M)": [380.0, -65.0, 315.0, -40.0, 210.0],
        }
    )

    commentary_df = pd.DataFrame(
        {
            "Section": [
                "Executive Overview",
                "Revenue & Segment Drivers",
                "Margin Expansion & Efficiency",
                "Risk Factors & Outlook",
            ],
            "Management Discussion & Analysis (MD&A)": [
                "Apex Cloud Technologies Inc. delivered record FY2026 Total Revenue of $1,250.0M (+27.6% YoY), driven by enterprise AI platform adoption and a 122% Net Dollar Retention rate.",
                "Enterprise Subscription Revenue reached $1,050.0M (+31% YoY), while Professional Services accounted for $200.0M (+12% YoY). North America contributed 68% of revenue and EMEA/APAC contributed 32%.",
                "Gross Margin expanded by 400 bps from 70.0% in FY2025 to 74.0% in FY2026 due to GPU cluster optimization. Operating Income surged 70.5% YoY to $375.0M (30.0% Operating Margin).",
                "Key risks include cloud infrastructure supply constraints, foreign exchange headwinds in EMEA, and enterprise procurement cycle elongation. FY2027 guidance targets $1,520M-$1,560M in revenue.",
            ],
        }
    )

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        income_df.to_excel(writer, sheet_name="Income_Statement", index=False)
        balance_df.to_excel(writer, sheet_name="Balance_Sheet", index=False)
        cashflow_df.to_excel(writer, sheet_name="Cash_Flow", index=False)
        commentary_df.to_excel(writer, sheet_name="MD&A_Commentary", index=False)

    return output.getvalue()
