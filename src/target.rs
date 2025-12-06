//    Copyright 2025 Karesis
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

use target_lexicon::{PointerWidth, Triple};

#[derive(Debug, Clone)]
pub struct TargetMetrics {
    pub triple: Triple,
    pub ptr_byte_width: u64,
    pub ptr_align: u64,
}

impl TargetMetrics {
    pub fn host() -> Self {
        Self::from_triple(Triple::host())
    }

    pub fn from_str(s: &str) -> Result<Self, String> {
        let triple: Triple = s
            .parse()
            .map_err(|e| format!("Invalid target triple: {}", e))?;
        Ok(Self::from_triple(triple))
    }

    fn from_triple(triple: Triple) -> Self {
        let width = match triple.pointer_width() {
            Ok(PointerWidth::U16) => 2,
            Ok(PointerWidth::U32) => 4,
            Ok(PointerWidth::U64) => 8,
            Err(_) => 8,
        };

        Self {
            triple,
            ptr_byte_width: width,
            ptr_align: width,
        }
    }

    pub fn usize_width(&self) -> u64 {
        self.ptr_byte_width
    }
}
