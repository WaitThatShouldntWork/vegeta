## Cyber domain pack (minimal)

- Sources: NVD, CPE, CISA KEV, EPSS, MITRE ATT&CK
- Nodes: CVE, Product, Vendor, Technique, Event
- Rels: AFFECTS, MITIGATES, RELATED_TO, REQUIRES, NEXT_IF
- Questions: ~12–15; start with exposure, criticality, controls

### Question shapes (examples)
- internet_exposed → requires asset node with edge to internet boundary; fallback to user slot
- affected_version_installed → compare asset’s installed CPEs vs vulnerable CPE ranges
- actively_exploited → KEV flag on CVE
- epss_high → EPSS score on CVE > threshold
- patch_available/workaround_available → vendor advisory retrieval (search trigger)


