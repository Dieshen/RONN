use ronn_onnx::ModelLoader;

#[test]
fn test_load_simple_model_onnx() {
    let fixture_path = "../../crates/ronn-api/tests/fixtures/simple_model.onnx";

    // Check if file exists
    if !std::path::Path::new(fixture_path).exists() {
        eprintln!("Fixture not found: {}", fixture_path);
        return;
    }

    let result = ModelLoader::load_from_file(fixture_path);

    match result {
        Ok(model) => {
            println!("Successfully loaded ONNX model!");
            println!("  IR version: {}", model.ir_version);
            println!("  Producer: {:?}", model.producer_name);
            println!("  Inputs: {}", model.inputs().len());
            println!("  Outputs: {}", model.outputs().len());
            println!("  Initializers: {}", model.initializers().len());

            for input in model.inputs() {
                println!("  Input: {} {:?} {:?}", input.name, input.shape, input.data_type);
            }

            for output in model.outputs() {
                println!("  Output: {} {:?} {:?}", output.name, output.shape, output.data_type);
            }
        }
        Err(e) => {
            panic!("Failed to load ONNX model: {}", e);
        }
    }
}

#[test]
fn test_load_add_model_onnx() {
    let fixture_path = "../../crates/ronn-api/tests/fixtures/add_model.onnx";

    // Check if file exists
    if !std::path::Path::new(fixture_path).exists() {
        eprintln!("Fixture not found: {}", fixture_path);
        return;
    }

    let result = ModelLoader::load_from_file(fixture_path);

    match result {
        Ok(model) => {
            println!("Successfully loaded add_model.onnx!");
            println!("  IR version: {}", model.ir_version);
            println!("  Inputs: {}", model.inputs().len());
            println!("  Outputs: {}", model.outputs().len());
        }
        Err(e) => {
            panic!("Failed to load add_model.onnx: {}", e);
        }
    }
}
