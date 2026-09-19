#include <cstdint>
#include <iostream>
#include <vector>

namespace {
bool read_exact(std::istream& input, char* data, std::size_t size) {
  input.read(data, static_cast<std::streamsize>(size));
  return input.gcount() == static_cast<std::streamsize>(size);
}

bool read_u32(std::istream& input, std::uint32_t& value) {
  unsigned char bytes[4];
  if (!read_exact(input, reinterpret_cast<char*>(bytes), sizeof(bytes))) return false;
  value = static_cast<std::uint32_t>(bytes[0]) |
          (static_cast<std::uint32_t>(bytes[1]) << 8) |
          (static_cast<std::uint32_t>(bytes[2]) << 16) |
          (static_cast<std::uint32_t>(bytes[3]) << 24);
  return true;
}

void write_u32(std::ostream& output, std::uint32_t value) {
  const unsigned char bytes[4] = {
      static_cast<unsigned char>(value & 0xff),
      static_cast<unsigned char>((value >> 8) & 0xff),
      static_cast<unsigned char>((value >> 16) & 0xff),
      static_cast<unsigned char>((value >> 24) & 0xff)};
  output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}
}  // namespace

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);

  // Protocol: width, height, RGB frame bytes; output is a 2x nearest frame.
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  while (read_u32(std::cin, width) && read_u32(std::cin, height)) {
    if (width == 0 || height == 0 || width > 16384 || height > 16384) return 2;
    const std::size_t input_size = static_cast<std::size_t>(width) * height * 3;
    std::vector<unsigned char> input(input_size);
    if (!read_exact(std::cin, reinterpret_cast<char*>(input.data()), input.size())) return 1;

    const std::uint32_t output_width = width * 2;
    const std::uint32_t output_height = height * 2;
    std::vector<unsigned char> output(static_cast<std::size_t>(output_width) * output_height * 3);
    for (std::uint32_t y = 0; y < output_height; ++y) {
      for (std::uint32_t x = 0; x < output_width; ++x) {
        const std::size_t source = (static_cast<std::size_t>(y / 2) * width + x / 2) * 3;
        const std::size_t target = (static_cast<std::size_t>(y) * output_width + x) * 3;
        output[target] = input[source];
        output[target + 1] = input[source + 1];
        output[target + 2] = input[source + 2];
      }
    }

    write_u32(std::cout, output_width);
    write_u32(std::cout, output_height);
    write_u32(std::cout, static_cast<std::uint32_t>(output.size()));
    std::cout.write(reinterpret_cast<const char*>(output.data()),
                    static_cast<std::streamsize>(output.size()));
    std::cout.flush();
  }
  return 0;
}
