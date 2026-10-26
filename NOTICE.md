# Notice on the data in this repository

I simulated all the data in this repository. I did not include third-party data, documents or models.

- **How I generated them.** I used my [nearshore wave simulator](tasks/nearshore-wave-forecast/solution/sim.py) and [data generator](tools/nearshore-wave-forecast/generation/make_data.py) to generate the observations, buoy, wind and tide records, chart, example conditions, and original three unseen cases. I added the final two unseen cases with [add_conditions.py](tools/nearshore-wave-forecast/hardening/add_conditions.py).
- **The site is fictitious.** I designed a fictitious coast; I did not model a real site, radar or survey.
