# Verified sources for the constructed microgrid benchmark

Official sources and current terms were accessed on 8 September 2026. The verification record is [source_verification.json](../provenance/source_verification.json). No source array was changed by this audit.

| Local file | Source identity and evidence type | Native calendar and representation | Verification |
|---|---|---|---|
| `annual_load_pattern_CAMX_baseline.csv` | Modeled CAMX large-office baseline load from NREL Data Catalog dataset 205 | 8,760 hourly rows; native timestamps are in 2023; `units=kWh`; not measured microgrid demand | Exact byte match to the official California ZIP member |
| `cambium_grid_data_California_cambium_grid_value.csv` | California grid-value scenario output, `StdScen20_MidCase`, distributed in dataset 205 | 8,760 hourly rows; native year and timestamps are 2022; not a retail electricity tariff | Exact byte match to the official California ZIP member |
| `cambium_grid_data_California_cambium_co2_rate_lrmer.csv` | California long-run marginal-emissions scenario output, `StdScen20_MidCase`, distributed in dataset 205 | 8,760 hourly rows; native year and timestamps are 2022; not measured contemporaneous site emissions | Exact byte match to the official California ZIP member |
| `pvwatts_pasadena_1kw.json` | PVWatts V8 modeled AC production for a 1 kW Pasadena system; API version 8.5.0 | 8,760 hourly AC outputs in W; NSRDB PSM V3 GOES `tmy-2020 3.2.0`, resource station 84341 | Fresh official API replay: 8,760/8,760 AC values exactly equal; maximum absolute difference 0 W |

The three CSVs are distributed with [Efficiency and Demand Flexibility in Large Office Buildings, dataset 205](https://data.nlr.gov/submissions/205), associated with the report by Joyce McLaren, Thomas Bowen and Chioke Harris, DOI [10.2172/1989231](https://doi.org/10.2172/1989231). Their official archive is [California-1679428758.zip](https://data.nlr.gov/system/files/205/California-1679428758.zip). Member names, retrieved archive hash, local/member hashes and native first/last records are preserved in the verification record.

PV inputs are preserved in that record and match the local JSON: latitude 34.1478, longitude -118.1445, capacity 1 kW, azimuth 180 degrees, tilt 20 degrees, array type 1, module type 1, losses 14%, DC/AC ratio 1.2, ground coverage ratio 0.4 and inverter efficiency 96%. See [PVWatts V8 API documentation](https://developer.nlr.gov/docs/solar/pvwatts/v8/). The resource grid location is approximately 2,031 m from the requested coordinate, at 34.130001/-118.139999, UTC-8. `TMY-2020` identifies the typical-meteorological-year resource version; it is not a claim that the complete series was observed during calendar year 2020. The 8,760-row comparison and the curated verification record are retained under `../provenance/`; the duplicate API response is omitted from this review package.

## Alignment and transformations

The simulator constructs a scenario by aligning positions in the four annual arrays by hour of year. It does not reconstruct contemporaneous operation at a real facility: the inputs have different native calendars, spatial aggregation and modeled meanings. The 8,760 hourly positions yield 365 daily episodes, each with 96 supervisory steps through fourfold hourly repetition (zero-order hold). Hourly source energy and power representations are handled by the loader; PV AC output is converted from W to kW before benchmark scaling.

This file documents raw source identity. Fitted normalization constants, training/validation/test boundaries and forecast construction must be read from the associated run configuration and preprocessing record. They are separate from the source metadata. Cost and carbon remain transformed benchmark scores unless a run explicitly establishes and documents a physical tariff/emissions calibration.

EV demand, EV availability, lighting occupancy, forecast perturbations and stress scenarios are generated assumptions. Battery size, efficiency, grid limits and dynamic coefficients are model parameters. These inputs and the source verification do not provide measured plant, hardware or independently calibrated converter evidence. Replaying an unperturbed profile slice is an evaluation on the same source family.

## Terms and attribution

The [dataset 205 license](https://data.nlr.gov/node/205/license) applies to the three verified CSV members. Its complete substantive notice is preserved as plain text at `../provenance/licenses/dataset205_license.txt`. It grants use/copy rights without a fee subject to retaining the complete notice in copies and crediting DOE/NREL/ALLIANCE in publications resulting from use. The data are provided as-is and do not imply source-provider endorsement. This notice applies to the three CSVs; the PVWatts terms are documented separately.

PVWatts is documented separately under the [NLR Developer Network terms](https://developer.nlr.gov/terms/), which state that the provided data and code may be used for any purpose. The complete terms are retained at `../provenance/licenses/pvwatts_developer_terms.txt`; the non-endorsement and as-is provisions remain applicable. Verification used the public service; no access credential is stored in this package.

Suggested publication credit: This study uses public modeled profiles supplied by DOE/NREL/ALLIANCE through the NREL Data Catalog and PVWatts/NSRDB. These organizations have not validated or endorsed the constructed microgrid benchmark or its findings.
