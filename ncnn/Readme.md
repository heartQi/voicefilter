# Install ncnn

```
  conan create conan/recipes/ncnn/all ncnn/1.0.1@mthor/stable -s os=Macos -k
  conan create conan/recipes/ncnn/all ncnn/1.0.1@mthor/stable -s compiler="Visual Studio" -s arch=x86_64
```

# Prepare dependency

```
mkdir build
cd build
conan install ../conan -s os=Macos -r mthor --update
cmake -GXcode ..
```