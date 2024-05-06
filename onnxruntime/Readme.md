https://github.com/microsoft/onnxruntime/releases/tag/v1.17.3

下载二进制版本，macos的话解压到macos目录，windows平台解压到windows目录

mkdir build
cd build
cmake .. -GXcode

Windows平台是

cmake .. -G"Visual Studio 17 2022"

就可以编译运行了。

