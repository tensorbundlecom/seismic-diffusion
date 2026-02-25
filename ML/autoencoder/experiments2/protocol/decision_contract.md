# Decision Contract

Bu dosya, anlam karmasasi olmamasi icin sabit kurallari tanimlar.

## Ana Kural

Kodlama, zorunlu kararlar dondurulmadan baslamaz.

## Tek Gercek Kaynak (Single Source of Truth)

- Karar kayitlari: `protocol/decisions.md`
- Deney spesifikasyonu: `protocol/experiment_001.md`
- Terim sozlugu: bu dosya

## Terim Sozlugu

- `Frozen`: Karar verildi, degismeyecek.
- `Pending`: Henuz karar verilmedi.
- `Revision`: Frozen karar degisti; gerekce zorunlu.

## Karar Yazim Formati

Her karar bu sablonla yazilir:

- `ID`: D001 gibi tekil kimlik
- `Question`: Karar sorusu
- `Options`: A/B/C secenekleri
- `Chosen`: Secilen secenek
- `Why`: Teknik gerekce
- `Impact`: Veri, model, train, eval etkisi
- `Status`: Pending/Frozen/Revision
- `Date`: YYYY-MM-DD

## Isleyis

1. Tek bir karar sorusu sorulur.
2. Secenekler net ve birbirini dislayan sekilde yazilir.
3. Sen secersin.
4. Karar `Frozen` edilir.
5. Sonraki karara gecilir.
