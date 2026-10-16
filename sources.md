# Sources and indicator definitions

Historical observations are retrieved live from the official World Bank Indicators API. The app does not substitute random or fabricated values when observations are missing.

API documentation: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation

API pattern:
`https://api.worldbank.org/v2/country/{ISO2}/indicator/{INDICATOR_CODE}?format=json&per_page=2000&date={START}:{END}`

Countries (ISO alpha-2): Mexico/MX, Brazil/BR, Argentina/AR, Colombia/CO, Chile/CL, Peru/PE, Venezuela/VE, Ecuador/EC, Uruguay/UY. Brazil is included as a Latin American comparison country.

| Indicator | World Bank code | Units shown | Definition / caution |
|---|---|---|---|
| Life expectancy | `SP.DYN.LE00.IN` | years | Life expectancy at birth, total (years). |
| Unemployment rate (national estimate) | `SL.UEM.TOTL.NE.ZS` | % of labor force | Unemployment, total (% of total labor force), national estimate. |
| GDP per capita | `NY.GDP.PCAP.CD` | current US$ per person | GDP divided by population; not average wealth or household income. |
| Crude birth rate | `SP.DYN.CBRT.IN` | births per 1,000 people per year | Annual live births per 1,000 people; not a total birth count. |
| Net migration | `SM.POP.NETM` | people (net total) | Net migration; not emigration alone. |
| Intentional homicide rate | `VC.IHR.PSRC.P5` | per 100,000 people | Intentional homicides per 100,000 population. |
| Consumer price inflation | `FP.CPI.TOTL.ZG` | % annual change | Annual consumer price index percentage change. |
| Central government debt | `GC.DOD.TOTL.GD.ZS` | % of GDP | Central government debt relative to GDP. |
| Total reserves including gold | `FI.RES.TOTL.CD` | current US$ | Total reserves including gold, current US dollars. |
| External balance on goods and services | `NE.RSB.GNFS.ZS` | % of GDP | External balance on goods and services as a share of GDP. |

Indicator metadata and observations can change as the World Bank revises its database. The app records the retrieval date and links to the series endpoint. It requests current year minus 70 through current year minus 1, inclusive, and reports actual coverage and usable observation count.
