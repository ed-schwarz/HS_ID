from encoder import Encoder, Algorithm, SymbolSize

encoder = Encoder(128, SymbolSize.G2x8, Algorithm.ReedSalomon)
encoder.generate()
collions = encoder.encode(255)
print(f"collions {collions}")
