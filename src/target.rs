use target_lexicon::{Triple, PointerWidth};

#[derive(Debug, Clone)]
pub struct TargetMetrics {
    pub triple: Triple,
    pub ptr_byte_width: u64,
    pub ptr_align: u64,
    // 未来可以加 endianness, simd_width 等
}

impl TargetMetrics {
    // 从 host (当前运行电脑) 获取默认配置
    pub fn host() -> Self {
        Self::from_triple(Triple::host())
    }

    // 从字符串解析 (e.g. "wasm32-unknown-unknown")
    pub fn from_str(s: &str) -> Result<Self, String> {
        let triple: Triple = s.parse().map_err(|e| format!("Invalid target triple: {}", e))?;
        Ok(Self::from_triple(triple))
    }

    fn from_triple(triple: Triple) -> Self {
        // 根据架构判断指针宽度
        let width = match triple.pointer_width() {
            Ok(PointerWidth::U16) => 2,
            Ok(PointerWidth::U32) => 4,
            Ok(PointerWidth::U64) => 8,
            Err(_) => 8, // 默认回退到 64 位
        };

        Self {
            triple,
            ptr_byte_width: width,
            ptr_align: width, // 通常指针的对齐要求等于其宽度
        }
    }
    
    pub fn usize_width(&self) -> u64 {
        self.ptr_byte_width
    }
}