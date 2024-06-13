mod point;

extern crate enterpolation;
extern crate glutin_window;
extern crate graphics;
extern crate ndarray;
extern crate ndarray_npy;
extern crate opengl_graphics;
extern crate piston;
extern crate soloud;

use std::env;

use soloud::*;

use point::Point;

use graphics::{Context, Transformed};

use enterpolation::bspline::BSpline;
use enterpolation::Curve;
use glutin_window::GlutinWindow;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::*;
use piston::input::{RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
use piston::window::WindowSettings;
use piston::EventLoop;

const NPY_SCALE: f64 = 50.0;

pub struct App {
    gl: GlGraphics,
    time: f64,
    ema: ndarray::Array2<f64>,
    wav: soloud::audio::Wav,
    articulators: Vec<Articulator>,
    show_control_points: bool,
    sl: soloud::Soloud,
}

#[derive(Debug, Clone)]
pub struct Articulator {
    name: String,
    point: Point,
}

impl Articulator {
    fn new(name: String, x: f64, y: f64) -> Articulator {
        Articulator {
            name,
            point: Point::new(x, y),
        }
    }

    fn update(&mut self, ema: &ndarray::Array2<f64>, time: f64) {
        match self.name.as_str() {
            "LI" => {
                self.point.x = ema[[time as usize, 0]] * NPY_SCALE;
                self.point.y = ema[[time as usize, 1]] * NPY_SCALE;
            }
            "UL" => {
                self.point.x = ema[[time as usize, 2]] * NPY_SCALE;
                self.point.y = ema[[time as usize, 3]] * NPY_SCALE;
            }
            "LL" => {
                self.point.x = ema[[time as usize, 4]] * NPY_SCALE;
                self.point.y = ema[[time as usize, 5]] * NPY_SCALE;
            }
            "TT" => {
                self.point.x = ema[[time as usize, 6]] * NPY_SCALE;
                self.point.y = ema[[time as usize, 7]] * NPY_SCALE;
            }
            "TB" => {
                self.point.x = ema[[time as usize, 8]] * NPY_SCALE;
                self.point.y = ema[[time as usize, 9]] * NPY_SCALE;
            }
            "TD" => {
                self.point.x = ema[[time as usize, 10]] * NPY_SCALE;
                self.point.y = ema[[time as usize, 11]] * NPY_SCALE;
            }
            _ => (),
        }
    }
}

impl App {
    fn new(ema: ndarray::Array2<f64>, wav: soloud::audio::Wav) -> App {
        let sl = soloud::Soloud::default().unwrap();
        sl.play(&wav);

        let mut articulators = vec![
            Articulator::new("LI".to_string(), 0.0, 0.0),
            Articulator::new("UL".to_string(), 0.0, 0.0),
            Articulator::new("LL".to_string(), 0.0, 0.0),
            Articulator::new("TT".to_string(), 0.0, 0.0),
            Articulator::new("TB".to_string(), 0.0, 0.0),
            Articulator::new("TD".to_string(), 0.0, 0.0),
        ];
        for articulator in &mut articulators {
            articulator.update(&ema, 0.0);
        }
        App {
            gl: GlGraphics::new(OpenGL::V3_2),
            articulators,
            time: 0.0,
            ema,
            wav,
            show_control_points: false,
            sl,
        }
    }

    fn render(&mut self, args: &RenderArgs) {
        const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
        const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
        const GREY: [f32; 4] = [0.5, 0.5, 0.5, 1.0];

        let (start_x, start_y): (f64, f64) = (400.0, 250.0);
        let articulators = &self.articulators;
        let show_control_points = self.show_control_points;

        self.gl.draw(args.viewport(), |c, gl| {
            graphics::clear(WHITE, gl);

            let mut control_points = Vec::new();

            // Show all non-tongue articulator points
            for articulator in articulators {
                if !articulator.name.contains("T") {
                    draw_keypoint(&articulator.point, BLACK, start_x, start_y, &c, gl);
                }
            }

            draw_tongue(articulators, &mut control_points, start_x, start_y, &c, gl);
            draw_palate(&mut control_points, start_x, start_y, &c, gl);
            draw_tongue_support(&mut control_points, start_x, start_y, &c, gl);

            if show_control_points {
                for point in &control_points {
                    draw_keypoint(&point, GREY, start_x, start_y, &c, gl);
                }
            }
        });
    }

    fn update(&mut self, _args: &UpdateArgs) {
        if self.time < self.ema.shape()[0] as f64 {
            for articulator in &mut self.articulators {
                articulator.update(&self.ema, self.time);
            }
            self.time += 1.0;
        } else {
            // self.time = 0.0;
        }
    }

    fn key_pressed(&mut self, key: piston::Button) {
        match key {
            piston::Button::Keyboard(piston::Key::Space) => {
                self.time = 0.0;
                self.sl.play(&self.wav);
            }

            piston::Button::Keyboard(piston::Key::C) => {
                self.show_control_points = !self.show_control_points;
            }
            _ => (),
        }
    }
}

fn draw_palate(
    control_points: &mut Vec<Point>,
    start_x: f64,
    start_y: f64,
    c: &Context,
    gl: &mut GlGraphics,
) {
    const PALATE_PINK: [f32; 4] = [1.0, 0.5, 0.5, 1.0];

    let base_x = -80.0;
    let base_y = 60.0;

    let points = Vec::from([
        Point::new(base_x + 30.0, base_y),
        Point::new(base_x - 100.0, base_y + 30.0),
        Point::new(base_x - 170.0, base_y + 10.0),
        Point::new(base_x - 180.0, base_y - 30.0),
        Point::new(base_x - 190.0, base_y - 50.0),
        Point::new(base_x - 186.0, base_y - 63.0),
        Point::new(base_x - 170.0, base_y - 55.0),
        // after velum
        Point::new(base_x - 130.0, base_y - 15.0),
        Point::new(base_x - 80.0, base_y - 20.0),
        Point::new(base_x - 20.0, base_y - 25.0),
        // front of palate
        Point::new(base_x - 0.0, base_y - 70.0),
        Point::new(base_x + 20.0, base_y - 90.0),
        Point::new(base_x + 25.0, base_y - 70.0),
        // front/top of palate
        Point::new(base_x + 10.0, base_y - 15.0),
        Point::new(base_x + 30.0, base_y),
        Point::new(base_x + 100.0, base_y - 40.0),
    ]);

    let weights = Vec::from([
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // after velum
        3.0, 1.0, 1.0, // front of palate
        1.0, 1.0, 1.0, // front/top of palate
        1.0, 0.5, 0.1,
    ]);

    control_points.extend(points.iter());

    interpolate_draw(&points, Some(weights), PALATE_PINK, start_x, start_y, c, gl);
}

fn draw_tongue_support(
    control_points: &mut Vec<Point>,
    start_x: f64,
    start_y: f64,
    c: &Context,
    gl: &mut GlGraphics,
) {
    const PALATE_PINK: [f32; 4] = [1.0, 0.5, 0.5, 1.0];

    let base_x = -100.0;
    let base_y = -60.0;

    let points = Vec::from([
        // track bottom of tongue
        Point::new(base_x - 180.0, base_y - 80.0),
        Point::new(base_x - 100.0, base_y - 120.0),
        Point::new(base_x - 10.0, base_y - 80.0),
        // bottom of support
        Point::new(base_x + 20.0, base_y - 40.0),
        Point::new(base_x + 10.0, base_y - 100.0),
        Point::new(base_x - 100.0, base_y - 130.0),
        Point::new(base_x - 200.0, base_y - 100.0),
        Point::new(base_x - 180.0, base_y - 80.0),
    ]);

    let weights = Vec::from([1.0, 1.0, 0.3, 1.0, 1.0, 4.0, 2.0, 1.0]);

    control_points.extend(points.iter());

    interpolate_draw(&points, Some(weights), PALATE_PINK, start_x, start_y, c, gl);
}

fn draw_tongue(
    articulators: &Vec<Articulator>,
    control_points: &mut Vec<Point>,
    start_x: f64,
    start_y: f64,
    c: &Context,
    gl: &mut GlGraphics,
) {
    const PINK: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

    let tt = &articulators[3].point;
    let tb = &articulators[4].point;
    let td = &articulators[5].point;

    let (points, weights) = tongue_points(*tt, *tb, *td);
    control_points.extend(points.iter());

    interpolate_draw(&points, Some(weights), PINK, start_x, start_y, c, gl);
}

fn interpolate_draw(
    points: &Vec<Point>,
    weights: Option<Vec<f64>>,
    color: [f32; 4],
    start_x: f64,
    start_y: f64,
    c: &Context,
    gl: &mut GlGraphics,
) {
    let interp_points = interpolate(points, weights);

    draw_curve(&interp_points, color, start_x, start_y, c, gl);
}

fn tongue_points(tt: Point, tb: Point, td: Point) -> (Vec<Point>, Vec<f64>) {
    let points = Vec::from([
        // top of tongue
        tt,
        Point::new(tt.x - 30.0, tb.y),
        tb,
        Point::new(tb.x - 40.0, tb.y),
        td,
        // back of tongue
        Point::new(td.x - 30.0, td.y - 50.0),
        // bottom of tongue
        Point::new(-280.0, -140.0),
        Point::new(-200.0, -180.0),
        Point::new(-110.0, -140.0),
        // front of tongue
        Point::new(tt.x - 10.0, tt.y - 60.0),
        Point::new(tt.x, tt.y - 20.0),
        Point::new(tt.x, tt.y),
    ]);

    let weights = Vec::from([1.0, 2.0, 5.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0]);

    (points, weights)
}

fn interpolate(points: &Vec<Point>, weights: Option<Vec<f64>>) -> Vec<Point> {
    let weights = weights.unwrap_or(vec![1.0; points.len()]);

    let points = points
        .iter()
        .zip(weights.iter())
        .map(|(point, weight)| (*point, *weight))
        .collect::<Vec<(Point, f64)>>();

    let first_point = points[0].0;

    let mut knots = Vec::new();

    for i in 0..(points.len() / 2) {
        knots.push(i as f64);
        knots.push(i as f64);
    }
    knots.push((points.len() / 2) as f64);

    let nurbs = BSpline::builder()
        .elements_with_weights(points)
        .knots(knots)
        .constant::<3>()
        .build()
        .unwrap();

    let mut interp_points: Vec<Point> = Vec::new();
    for (_, point) in nurbs.take(100).enumerate() {
        // println!("{}, {}", point.x, point.y);
        if f64::is_finite(point.x) && f64::is_finite(point.y) {
            interp_points.push(point);
        }
    }

    interp_points.push(first_point);

    interp_points
}

fn draw_curve(
    points: &Vec<Point>,
    color: [f32; 4],
    start_x: f64,
    start_y: f64,
    c: &Context,
    gl: &mut GlGraphics,
) {
    for i in 0..points.len() - 1 {
        graphics::line(
            color,
            1.0,
            [
                start_x + points[i].x,
                start_y - points[i].y,
                start_x + points[i + 1].x,
                start_y - points[i + 1].y,
            ],
            c.transform,
            gl,
        );
    }
}

fn draw_keypoint(
    point: &Point,
    color: [f32; 4],
    start_x: f64,
    start_y: f64,
    c: &Context,
    gl: &mut GlGraphics,
) {
    let transform = c.transform.trans(start_x, start_y).trans(point.x, -point.y);
    graphics::rectangle(
        color,
        graphics::rectangle::square(0.0, 0.0, 5.0),
        transform,
        gl,
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        println!("Usage: <path-to-npy> <path-to-wav>");
        std::process::exit(1);
    }

    let npy_path = &args[1];
    let wav_path = std::path::Path::new(&args[2]);

    let venture: ndarray::Array2<f64> = ndarray_npy::read_npy(npy_path).unwrap();

    // let weight = 2.0f64.sqrt() / 2.0;

    // ORDER OF ARTICULATORS:
    // LI XY, UL XY, LL XY, TT XY, TB XY, TD XY

    let opengl = OpenGL::V3_2;

    let mut window: GlutinWindow = WindowSettings::new("vocal-tract", [600, 600])
        .graphics_api(opengl)
        .exit_on_esc(true)
        .build()
        .unwrap();

    // let sl = soloud::Soloud::default().unwrap();
    let mut wav = soloud::audio::Wav::default();
    wav.load(wav_path).unwrap();
    // sl.play(&wav);

    let mut app = App::new(venture, wav);

    let mut events = Events::new(EventSettings::new());
    events.set_ups(50);
    while let Some(e) = events.next(&mut window) {
        if let Some(args) = e.render_args() {
            app.render(&args);
        }

        if let Some(args) = e.update_args() {
            app.update(&args);
        }

        if let Some(Button::Keyboard(key)) = e.press_args() {
            app.key_pressed(Button::Keyboard(key));
        }
    }
}
