# Setup

```shell
wget -q https://git.io/J0fjL -O IAM_Words.zip
unzip -qq IAM_Words.zip
mkdir -p resources/datasets/IAM_Words/words
mkdir -p resources/cache
tar -xf IAM_Words/words.tgz -C datasets/IAM_Words//words
mv IAM_Words/words.txt datasets/IAM_Words
rm -r IAM_Words
```
