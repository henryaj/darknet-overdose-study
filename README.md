# Does shutting down dark web markets kill people? (2026)

_[Read the full paper](https://cdn.jsdelivr.net/gh/henryaj/darknet-overdose-study/paper/main.pdf)_.

In July 2017, the FBI and Europol pulled off a pretty stunning takedown of two darknet marketplaces. These marketplaces are a bit like eBay but for any illicit substance you could think of; see Gwern's post for more on them. They use Tor hidden services to evade detection - unlike normal websites, the operator's IP address is hidden by routing traffic through multiple encrypted layers.

Uncovering the owners of these websites usually requires good old-fashioned detective work - Ross Ulbricht of Silk Road fame was famously caught out when the first forum post mentioning the site turned out to be from him, and he was traced through the email address he used.

In Operation Bayonet, AlphaBay, run by a Canadian called Alexandre Cazes, was taken offline when early emails sent by the site had a reply-to address belonging to the operator of the site. But simultanously, Dutch authorities had seized a second marketplace, Hansa - and instead of taking it offline, they took control of it, updating its source code to collect data on users and their transactions. Right when AlphaBay went offline, users flocked to Hansa, playing into the hands of law enforcement.

After a few weeks, Hansa was taken offline as well. This led to a void which would eventually be filled by more marketplaces (cf. the war on drugs largely being a waste of time). My guess was that a good portion of users would simply switch to street dealers. This seems robustly bad; buying online means you get the benefit of buying from a regular seller with a brand rep to protect, and users can read reviews of the product they're buying, as well as avoiding rip-off or violent encounters with dealers.

I used Claude Code to analyse fentanyl overdoses in the months after the takedowns to see if there was a discontinuity, my hypothesis being that ODs would jump as users were pushed to buy adulterated product from local dealers.

Specifically, I looked at CDC data on overdoses from 2015-2019, using an interrupted time series analysis to check for a discontinuity in July 2017, around the time of the takedowns. I specifically focused on synthetic opioids as these are the main driver of overdose deaths.

The findings were surprising:

- Deaths actually decelerated after the takedown, which is the opposite of what I and harm reduction advocates would have predicted
- Synthetic opioid deaths grew 677 per month pre-takedown and 291 per month post-takedown, a 57% reduction in the growth rate
- Causation is murky: running "placebo tests" to check whether the same thing happened at different dates found similar decelerations in Jan 2017, Jan 2018, suggesting a broader inflection point at the time. The takedowns may have contributed or just coincided 

Basically, [Betteridge's law](https://en.wikipedia.org/wiki/Betteridge%27s_law_of_headlines) applies.

Some limitations:

- National aggregate data is a blunt instrument - we can't stratify by parts of the US that have high or low darknet marketplace usage
- Using 12 month rolling averages smooths out short-term spikes
- Vendors may have simply migrated to other markets more quickly than I'm imagining

Zambiasi (2022) found street drug crimes spiked for ~18 days after darknet shutdowns, then returned to normal (which seems unlikely to me - why 18 days?).

Nonetheless, the "whack-a-mole" problem persists: shut down one marketplace, users move to another. DrugHub seems to be the most popular one at the moment - the dark web is very much still alive.

## Citation

If you use this work, please cite:

```bibtex
@misc{stanley2025darknet,
  author = {Stanley, Henry},
  title = {Does Shutting Down Dark Web Markets Kill People?},
  year = {2025},
  howpublished = {\url{https://github.com/henryaj/darknet-overdose-study}},
}
```
