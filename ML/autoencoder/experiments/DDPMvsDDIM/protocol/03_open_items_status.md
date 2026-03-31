# Open Items Status

Bu belge, onceki tartismalarda acik kalan maddelerin durumunu kapatmak icin tutulur.

## Closed in v1

- event-wise split politikasi: `CLOSED`
- rare large-event handling: `CLOSED`
- Stage-1 training policy: `CLOSED`
- latent target (`mu`): `CLOSED`
- latent sanity gate: `CLOSED`
- main metrics (`spec_corr`, `LSD`, `MR-LSD`): `CLOSED`
- sampler fairness protocol: `CLOSED`
- runtime defaults: `CLOSED`
- seed policy: `CLOSED`

## Implementation Closure

Frozen `v1` icin asagidaki maddeler uygulanmistir:

- event-wise split generator `DDPMvsDDIM` kutusuna baglandi
- Stage-1 event-wise training tamamlandi
- latent sanity gate eklendi ve gecerli checkpoint ile calisti
- `LSD` ve `MR-LSD` frozen metric path'i kullanildi
- sampler code path'i same-initial-noise kuralina cekildi
- `DDPM` ic adim gurultusu de deterministik hale getirildi
- metrics-only full evaluation ve visual subset evaluation ayrildi

## Remaining Work

Bu noktadan sonra kalan isler kavramsal degil, improvement fazidir:

- diffusion denoiser iyilestirmeleri
- yeni varyantlarin ayni metriklerle tekrar karsilastirilmasi
- sonuclarin `docs/improvement_tracking.md` icinde birikimli takibi
