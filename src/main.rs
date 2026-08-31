use crate::image::Image;

#[path = "image_processing/image.rs"]
pub mod image;

#[path = "image_processing/scanner.rs"]
pub mod scanner;

#[path = "neural_network/perceptron.rs"]
pub mod perceptron;

fn main() {
    let test_path: String =
        "C:\\Users\\uxbei\\Desktop\\Rust\\Number-Guesser-Rust\\src\\images\\test_3.bmp".to_string();
    let image: Image;

    match image::load_data(test_path) {
        Ok(data) => {
            println!("Image loaded successfully");
            image = data;
            println!("The number is {}", image.number);
        }
        Err(err) => {
            eprintln!("Error occured: {}", err)
        }
    }
}
