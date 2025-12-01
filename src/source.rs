/// ======================================================
/// 1. 基础类型: Span 与FileId
/// ======================================================
use std::ops::Range;

/// 简单的文件句柄，避免到处传 String
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileId(pub usize);

/// 源码区间：只存偏移量，不存内容，Copy 代价极小
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// 方便转换为 Range 用于切片： &src[span.as_range()]
    pub fn as_range(&self) -> Range<usize> {
        self.start..self.end
    }
}

/// ======================================================
/// 2. 单个文件结构: SourceFile
/// ======================================================

#[derive(Debug)]
pub struct SourceFile {
    /// 文件名 (用于报错显示)
    pub name: String,
    /// 完整源码
    pub src: String,
    /// 每一行开始的字节偏移量缓存
    /// lines[0] = 0, lines[1] = 第一行结束位置+1...
    pub line_starts: Vec<usize>,
}

impl SourceFile {
    pub fn new(name: String, src: String) -> Self {
        let line_starts = std::iter::once(0)
            .chain(src.match_indices('\n').map(|(i, _)| i + 1))
            .collect();

        Self {
            name,
            src,
            line_starts,
        }
    }

    /// 核心算法：利用二分查找快速定位行号
    pub fn lookup_line(&self, offset: usize) -> usize {
        match self.line_starts.binary_search(&offset) {
            Ok(line) => line + 1, // 精确匹配行首
            Err(line) => line,    // 在两行之间，binary_search 返回它“应该插入的位置”
        }
    }

    /// 获取 (Line, Column)
    /// 注意：编辑器通常行号列号都从 1 开始
    pub fn lookup_location(&self, offset: usize) -> (usize, usize) {
        let line = self.lookup_line(offset);
        let line_start = self.line_starts[line - 1];
        let col = offset - line_start + 1;
        (line, col)
    }
}

use std::fs;
/// ======================================================
/// 3. 总管：SourceManager
/// ======================================================
use std::io;
use std::path::Path;

#[derive(Debug)]
pub struct SourceManager {
    files: Vec<SourceFile>,
}

impl SourceManager {
    pub fn new() -> Self {
        Self { files: Vec::new() }
    }

    /// 加载文件并返回 ID
    pub fn load_file<P: AsRef<Path>>(&mut self, path: P) -> io::Result<FileId> {
        let src = fs::read_to_string(path.as_ref())?;
        let name = path.as_ref().to_string_lossy().to_string(); //? 这里为什么要用to_string_lossy?为了兼容？

        self.add_file(name, src)
    }

    /// 直接添加字符串（用于测试或 REPL）
    pub fn add_file(&mut self, name: String, src: String) -> io::Result<FileId> {
        let file = SourceFile::new(name, src);
        let id = FileId(self.files.len());
        self.files.push(file);
        Ok(id)
    }

    /// 获取文件引用
    pub fn get_file(&self, id: FileId) -> &SourceFile {
        &self.files[id.0]
    }

    /// 获取源码片段
    pub fn get_source(&self, id: FileId, span: Span) -> &str {
        &self.files[id.0].src[span.as_range()]
    }
}
