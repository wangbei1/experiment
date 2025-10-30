huggingface-cli download   kiwhansong/DFoT   --include "datasets/*"   --local-dir data   --local-dir-use-symlinks False

mv data/datasets/RealEstate10K_Full.tar.gz.part-ac data/
mv data/datasets/RealEstate10K_Full.tar.gz.part-aa data/
mv data/datasets/RealEstate10K_Full.tar.gz.part-ab data/

cat data/datasets/RealEstate10K_Full.tar.gz.part-aa \
    data/datasets/RealEstate10K_Full.tar.gz.part-ab \
    data/datasets/RealEstate10K_Full.tar.gz.part-ac \
    > data/datasets/RealEstate10K_Full.tar.gz

tar -xvzf data/datasets/RealEstate10K_Full.tar.gz -C data
