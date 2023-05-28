use criterion::*;
use encoder::{Algorithm, Encoder, SymbolSize};

fn walk_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("troughput");
    let size = 16 * (u16::MAX as usize);
    group.throughput(criterion::Throughput::Bytes((size * 2).try_into().unwrap()));
    let mut encoder: Encoder = Encoder::new(size, false, SymbolSize::G2x8, Algorithm::ReedSalomon);
    encoder.generate();
    group.bench_function("RS encode GF(2^8)", move |b| b.iter(|| black_box(encoder.encode(1))));
    group.throughput(criterion::Throughput::Bytes((size * 2 * 2).try_into().unwrap()));
    let mut encoder: Encoder = Encoder::new(size, false, SymbolSize::G2x16, Algorithm::ReedSalomon);
    encoder.generate();
    group.bench_function("RS encode GF(2^16)", move |b| b.iter(|| black_box(encoder.encode(1))));
    group.throughput(criterion::Throughput::Bytes((size * 2 * 4).try_into().unwrap()));
    let mut encoder: Encoder = Encoder::new(size, false, SymbolSize::G2x32, Algorithm::ReedSalomon);
    encoder.generate();
    group.bench_function("RS encode GF(2^32)", move |b| b.iter(|| black_box(encoder.encode(1))));
    let mut encoder: Encoder = Encoder::new(size, true, SymbolSize::G2x32, Algorithm::ReedSalomon);
    encoder.generate();
    group.bench_function("RS encode fast GF(2^32)", move |b| {
        b.iter(|| black_box(encoder.encode(1)))
    });
    let mut encoder: Encoder = Encoder::new(size, true, SymbolSize::G2x32, Algorithm::Sha1);
    encoder.generate();
    group.bench_function("Sha1 encode", move |b| b.iter(|| black_box(encoder.encode(1))));
    let mut encoder: Encoder = Encoder::new(size, true, SymbolSize::G2x32, Algorithm::Sha2);
    encoder.generate();
    group.bench_function("Sha2 encode", move |b| b.iter(|| black_box(encoder.encode(1))));
    let mut encoder: Encoder = Encoder::new(size, true, SymbolSize::G2x32, Algorithm::NoCode);
    encoder.generate();
    group.bench_function("No Code", move |b| b.iter(|| black_box(encoder.encode(1))));
}

criterion_group! {
  name = benches;
  config = Criterion::default().sample_size(10);
  targets = walk_benches
}

criterion_main!(benches);
