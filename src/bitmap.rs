use roaring::RoaringTreemap;

pub type Bitmap = RoaringTreemap;
pub type BitmapInt = u64;
pub type BitmapIter<'a> = roaring::treemap::Iter<'a>;
