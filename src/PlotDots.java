import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;

public class PlotDots extends JPanel {
    private double[] xValues, xZValues;
    private double[] yValues, yZValues;
    private int width = 1000;
    private int height = 1000;

    private double zoom = 1.0;
    private double panX = 0, panY = 0;
    private double centerX = 0, centerY = 0;
    private int lastMouseX, lastMouseY;

    private Electron electron;
    public JFrame frame;

    public PlotDots(Electron electron) {
        this.electron = electron;
        int points = electron.electronStateCamera.size();
        xValues = new double[points];
        xZValues = new double[points];
        yValues = new double[points];
        yZValues = new double[points];

        int i = 0;
        for (double[] state : electron.electronStateCamera) {
            xValues[i] = state[Electron.QX] * PhysicalData.zitterRadius;
            yValues[i] = state[Electron.QZ] * PhysicalData.zitterRadius;
            xZValues[i] = state[Electron.RX] * PhysicalData.zitterRadius;
            yZValues[i] = state[Electron.RZ] * PhysicalData.zitterRadius;
            i++;
        }

        // Center on origin (0,0) so atoms and trajectories are visible

        setBackground(new Color(20, 20, 30));

        // Title bar with key info
        String title = String.format("ELektron2 | E=%.0feV | Angle=%d\u00B0 | Spin=%+d",
                electron.getKineticEnergy(), (int) electron.getAngle(), PhysicalData.spin);
        frame = new JFrame(title);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.add(this);
        frame.setSize(width, height);
        frame.setVisible(true);

        // Scroll wheel zoom
        addMouseWheelListener(new MouseWheelListener() {
            @Override
            public void mouseWheelMoved(MouseWheelEvent e) {
                double factor = (e.getWheelRotation() < 0) ? 1.15 : 1.0 / 1.15;
                zoom *= factor;
                repaint();
            }
        });

        // Click and drag to pan
        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                lastMouseX = e.getX();
                lastMouseY = e.getY();
            }
        });
        addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                panX += e.getX() - lastMouseX;
                panY += e.getY() - lastMouseY;
                lastMouseX = e.getX();
                lastMouseY = e.getY();
                repaint();
            }
        });
    }

    // Map physical coordinate to pixel, with zoom + pan
    private int toScreenX(double physX) {
        double w = getWidth();
        double range = autoRange();
        double norm = ((physX - centerX) + range) / (2 * range); // 0..1 centered on trajectory
        return (int) (panX + w / 2 + (norm - 0.5) * w * zoom);
    }

    private int toScreenY(double physY) {
        double h = getHeight();
        double range = autoRange();
        double norm = ((physY - centerY) + range) / (2 * range);
        return (int) (panY + h / 2 - (norm - 0.5) * h * zoom); // flip Y
    }

    private double autoRange() {
        double maxVal = 1e-14;
        for (int i = 0; i < xValues.length; i++) {
            maxVal = Math.max(maxVal, Math.abs(xValues[i]));
            maxVal = Math.max(maxVal, Math.abs(yValues[i]));
            maxVal = Math.max(maxVal, Math.abs(xZValues[i]));
            maxVal = Math.max(maxVal, Math.abs(yZValues[i]));
        }
        // Include atom positions for full chain visibility
        for (int k = 0; k < PhysicalData.atomCount; k++) {
            maxVal = Math.max(maxVal, Math.abs(PhysicalData.atomZ[k] * PhysicalData.zitterRadius));
        }
        return maxVal * 1.1; // 10% margin
    }

    @Override
    protected void paintComponent(Graphics g) {
        if (xValues == null || xValues.length < 2) return;
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int w = getWidth();
        int h = getHeight();

        // Crosshairs at nucleus
        g2d.setColor(new Color(60, 60, 80));
        g2d.setStroke(new BasicStroke(1));
        int cx = toScreenX(0), cy = toScreenY(0);
        g2d.drawLine(0, cy, w, cy);
        g2d.drawLine(cx, 0, cx, h);

        // Draw atom chain: Bohr radius circle + nucleus dot at each atom
        double bohrMeters = PhysicalData.bohrRadius;
        int bohrW = Math.abs(toScreenX(bohrMeters) - toScreenX(-bohrMeters));
        g2d.setStroke(new BasicStroke(0.5f));
        for (int k = 0; k < PhysicalData.atomCount; k++) {
            double atomPhysZ = PhysicalData.atomZ[k] * PhysicalData.zitterRadius;
            int ax = toScreenX(0);
            int ay = toScreenY(atomPhysZ);

            // Bohr radius circle (faint)
            g2d.setColor(new Color(80, 80, 50, 60));
            g2d.drawOval(ax - bohrW / 2, ay - bohrW / 2, bohrW, bohrW);

            // Atom nucleus dot
            g2d.setColor(new Color(100, 255, 100, 200));
            g2d.fill(new Ellipse2D.Double(ax - 3, ay - 3, 6, 6));
        }
        g2d.setStroke(new BasicStroke(1));

        // Draw trajectory with color gradient
        int n = xValues.length;
        int renderLine = 0;
        for (int i = 0; i < n - 1; i++) {
            float t = (float) i / (float) (n - 1);

            int sx1 = toScreenX(xValues[i]);
            int sy1 = toScreenY(yValues[i]);
            int sx2 = toScreenX(xZValues[i]);
            int sy2 = toScreenY(yZValues[i]);

            if (sx1 < -100 || sx1 > w + 100 || sy1 < -100 || sy1 > h + 100) continue;

            // Mass center: yellow -> red gradient
            Color massColor = new Color(
                    255,
                    (int) (255 * (1 - t)),
                    0,
                    200
            );
            g2d.setColor(massColor);
            g2d.fill(new Ellipse2D.Double(sx1 - 1.5, sy1 - 1.5, 3, 3));

            // Charge center: cyan -> blue gradient
            Color chargeColor = new Color(
                    0,
                    (int) (255 * (1 - t)),
                    255,
                    200
            );
            g2d.setColor(chargeColor);
            g2d.fill(new Ellipse2D.Double(sx2 - 1.5, sy2 - 1.5, 3, 3));

            // Connecting line every 5th point
            renderLine++;
            if (renderLine == 5) {
                renderLine = 0;
                g2d.setColor(new Color(100, 100, 100, 80));
                g2d.draw(new Line2D.Double(sx1, sy1, sx2, sy2));
            }
        }

        // HUD: dark semi-transparent background for text
        g2d.setColor(new Color(0, 0, 0, 180));
        g2d.fillRect(5, 5, w - 10, 80);

        // Info text
        g2d.setFont(new Font("Monospaced", Font.PLAIN, 11));
        g2d.setColor(new Color(180, 180, 180));
        g2d.drawString("PARAMS | rangeMin: " + PhysicalData.rangeMin + " | rangeMax: " + PhysicalData.rangeMax, 10, 22);
        g2d.setColor(new Color(200, 200, 200));
        g2d.drawString(electron.getEXIT(), 10, 38);
        g2d.setColor(new Color(160, 160, 160));
        g2d.drawString(electron.printState(), 10, 54);

        // Energy out in bright orange
        g2d.setColor(new Color(255, 140, 0));
        g2d.setFont(new Font("SansSerif", Font.BOLD, 18));
        String energyOut = "Energy OUT: " + electron.format(electron.getKineticEnergy()) + " eV";
        g2d.drawString(energyOut, 10, h - 60);

        // Angle out
        g2d.setColor(new Color(100, 255, 100));
        g2d.setFont(new Font("SansSerif", Font.BOLD, 16));
        g2d.drawString("Angle: " + (int) electron.getAngle() + "\u00B0", 10, h - 35);

        // Legend
        int lx = w - 220, ly = h - 70;
        g2d.setColor(new Color(0, 0, 0, 160));
        g2d.fillRoundRect(lx - 5, ly - 5, 215, 55, 8, 8);
        g2d.setFont(new Font("SansSerif", Font.PLAIN, 12));

        g2d.setColor(new Color(255, 200, 0));
        g2d.fillOval(lx, ly + 2, 10, 10);
        g2d.setColor(new Color(200, 200, 200));
        g2d.drawString("Mass center (q)", lx + 16, ly + 12);

        g2d.setColor(new Color(0, 150, 255));
        g2d.fillOval(lx, ly + 20, 10, 10);
        g2d.setColor(new Color(200, 200, 200));
        g2d.drawString("Charge center (r)", lx + 16, ly + 30);

        g2d.setColor(new Color(100, 255, 100));
        g2d.fillOval(lx, ly + 38, 10, 10);
        g2d.setColor(new Color(200, 200, 200));
        g2d.drawString("C atoms (" + PhysicalData.atomCount + ") + Bohr r", lx + 16, ly + 48);

        // Zoom indicator
        g2d.setColor(new Color(120, 120, 120));
        g2d.setFont(new Font("SansSerif", Font.PLAIN, 10));
        g2d.drawString(String.format("Zoom: %.1fx  (scroll to zoom, drag to pan)", zoom), 10, h - 10);
    }
}
