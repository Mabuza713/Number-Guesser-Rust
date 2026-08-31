use std::io::{self};

pub struct Image {
    pub matrix: Vec<Vec<f64>>,
    pub number: u8,
}
impl Image {
    pub fn flatten_matrix(&self) -> Vec<f64> {
        let mut result: Vec<f64> = Vec::new();
        for row in &self.matrix {
            for &pixel in row {
                result.push(pixel);
            }
        }

        result
    }
}

fn parse_bytes_to_matrix(bytes_vec: &[u8]) -> Vec<Vec<f64>> {
    let mut matrix: Vec<Vec<f64>> = Vec::new();

    // in .bmp files
    // 18-21 bajt is width
    // 22-25 is height
    // 10-14 to offset pikseli
    //
    // If we do just &bytes_vec[18..22]; its a memory reference to a slice
    // we have no idea what it is gonna be, we just say to that there will be
    // 4 indexes to check
    let width_bytes: [u8; 4] = bytes_vec[18..22]
        .try_into()
        .expect("Array length is not right");
    let height_bytes: [u8; 4] = bytes_vec[22..26]
        .try_into()
        .expect("Array length is not right");

    let offset_bytes: [u8; 4] = bytes_vec[10..14]
        .try_into()
        .expect("Array length is not righr");

    let width = u32::from_le_bytes(width_bytes) as usize;
    let height = u32::from_le_bytes(height_bytes) as usize;
    let offset = u32::from_le_bytes(offset_bytes) as usize;

    let row_stride = (width * 3 + 3).div_ceil(4);

    println!("Parsing picture of size {}x{}", width, height);

    for x in (0..height).rev() {
        let mut row: Vec<f64> = Vec::with_capacity(width);
        let row_start = offset + x * row_stride;

        for y in 0..width {
            let px = row_start + y * 3;

            let b: u8 = bytes_vec[px];
            let g: u8 = bytes_vec[px + 1];
            let r: u8 = bytes_vec[px + 2];

            // I need to parse it to u16 cuz in rust it adds u8's which overflows
            let grey = ((b as u16 + g as u16 + r as u16) / 3) as f64;
            row.push(grey);
        }
        println!("{:?}", row);
        matrix.push(row);
    }
    println!("done");

    matrix
}

pub fn load_data(image_path: String) -> Result<Image, io::Error> {
    // let bytes: Vec<f64> = match std::fs::read(image_path) {
    //     Ok(data) => data,
    //     Err(err) => return Err(err),
    // };
    // This is the same things i guess still leave it cuz learning
    let bytes: Vec<u8> = std::fs::read(image_path)?;
    let matrix: Vec<Vec<f64>> = parse_bytes_to_matrix(&bytes);
    println!("[TEST] loaded {} of bytes.", bytes.len());

    let temp_number: u8 = 2;

    let result = Image {
        matrix,
        number: temp_number,
    };
    println!("Number: {}", result.number);

    Ok(result)
}
