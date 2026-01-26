import { Injectable, Logger } from '@nestjs/common';
import { Pool } from 'pg';
import { ConfigService } from '@nestjs/config';

export interface SlotState {
  name: string;
  phone: string;
  dob: string;
  visit_type: string;
  specialty: string;
  doctor_name: string;
  date: string;
  time: string;
  no_marketing: boolean | null;
}

export interface BookingResult {
  success: boolean;
  message: string;
  doctorName?: string;
  start?: string;
  end?: string;
  alternatives?: { start: string; end: string }[];
}

@Injectable()
export class VaBookingService {
  private readonly logger = new Logger(VaBookingService.name);
  private readonly pool: Pool | null;

  constructor(private readonly config: ConfigService) {
    const url = this.config.get<string>('DATABASE_URL');
    this.pool = url ? new Pool({ connectionString: url }) : null;
  }

  private ready(slots: SlotState): boolean {
    return (
      slots.name.trim() !== '' &&
      slots.phone.trim() !== '' &&
      slots.dob.trim() !== '' &&
      (slots.doctor_name.trim() !== '' || slots.specialty.trim() !== '') &&
      slots.date.trim() !== '' &&
      slots.time.trim() !== ''
    );
  }

  async tryBook(slots: SlotState, sessionId: string): Promise<BookingResult> {
    if (!this.pool) return { success: false, message: 'DB not configured' };
    if (!this.ready(slots))
      return { success: false, message: 'Slots incomplete' };

    const client = await this.pool.connect();
    try {
      // Resolve doctor
      let doctorRow;
      if (slots.doctor_name.trim() !== '') {
        const { rows } = await client.query(
          'SELECT * FROM doctors WHERE name LIKE $1 LIMIT 1',
          [`%${slots.doctor_name.trim()}%`],
        );
        doctorRow = rows[0];
      }
      if (!doctorRow && slots.specialty.trim() !== '') {
        const { rows } = await client.query(
          'SELECT * FROM doctors WHERE specialty LIKE $1 LIMIT 1',
          [`%${slots.specialty.trim()}%`],
        );
        doctorRow = rows[0];
      }
      if (!doctorRow) {
        return {
          success: false,
          message: 'لا يوجد طبيب متاح لهذا التخصص الآن',
        };
      }
      const doctorId = doctorRow.id as string;

      // Build datetime
      const start = this.combineDateTime(slots.date, slots.time);
      if (!start) return { success: false, message: 'تاريخ/وقت غير صالح' };
      const end = new Date(start.getTime() + 30 * 60 * 1000);

      // Check schedule
      const dow = start.getUTCDay(); // 0=Sunday
      const { rows: sched } = await client.query(
        'SELECT * FROM doctor_schedules WHERE doctor_id=$1 AND day_of_week=$2',
        [doctorId, dow],
      );
      const withinSchedule = sched.some((s) => {
        const [sh, sm] = (s.start_time as string).split(':').map(Number);
        const [eh, em] = (s.end_time as string).split(':').map(Number);
        const startMinutes = start.getUTCHours() * 60 + start.getUTCMinutes();
        const endMinutes = end.getUTCHours() * 60 + end.getUTCMinutes();
        const schedStart = sh * 60 + sm;
        const schedEnd = eh * 60 + em;
        return startMinutes >= schedStart && endMinutes <= schedEnd;
      });
      if (!withinSchedule) {
        const alternatives = this.suggestAlternatives(start, sched);
        return {
          success: false,
          message: 'الوقت خارج مواعيد الطبيب',
          alternatives,
        };
      }

      // Check conflicts
      const { rows: conflicts } = await client.query(
        'SELECT 1 FROM appointments WHERE doctor_id=$1 AND status=$2 AND start_datetime < $3 AND end_datetime > $4',
        [doctorId, 'booked', end.toISOString(), start.toISOString()],
      );
      if (conflicts.length > 0) {
        const alternatives = this.suggestAlternatives(start, sched);
        return {
          success: false,
          message: 'الوقت محجوز',
          alternatives,
        };
      }

      // Insert appointment
      await client.query(
        `INSERT INTO appointments
        (doctor_id, patient_name, patient_phone, patient_dob, visit_type, specialty, start_datetime, end_datetime, no_marketing, status, session_id)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)`,
        [
          doctorId,
          slots.name.trim(),
          slots.phone.trim(),
          slots.dob.trim(),
          slots.visit_type.trim(),
          slots.specialty.trim() || doctorRow.specialty,
          start.toISOString(),
          end.toISOString(),
          slots.no_marketing === true,
          'booked',
          sessionId,
        ],
      );

      return {
        success: true,
        message: 'تم الحجز بنجاح',
        doctorName: doctorRow.name,
        start: start.toISOString(),
        end: end.toISOString(),
      };
    } finally {
      client.release();
    }
  }

  private combineDateTime(dateStr: string, timeStr: string): Date | null {
    try {
      const isoDate = this.normalizeDate(dateStr.trim());
      if (!isoDate) return null;
      const time = timeStr.trim();
      const parts = time.split(':');
      let hour = 0;
      let minute = 0;
      if (parts.length >= 2) {
        hour = parseInt(parts[0], 10);
        minute = parseInt(parts[1], 10) || 0;
      } else {
        // fallback if time like "بين ٥ و٧" -> pick 17:00
        const digits = time.match(/\\d+/g);
        if (digits && digits.length > 0) {
          hour = parseInt(digits[0], 10);
        }
      }
      const d = new Date(
        `${isoDate}T${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}:00Z`,
      );
      return isNaN(d.getTime()) ? null : d;
    } catch {
      return null;
    }
  }

  private normalizeDate(dateStr: string): string | null {
    const parts = dateStr.split(/[\\/\\-]/);
    if (parts.length === 3) {
      const [d, m, y] = parts.map((p) => parseInt(p, 10));
      const year = y < 100 ? 2000 + y : y;
      return `${year}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
    }
    const asDate = new Date(dateStr);
    if (!isNaN(asDate.getTime())) {
      return asDate.toISOString().slice(0, 10);
    }
    return null;
  }

  private suggestAlternatives(
    start: Date,
    schedules: any[],
  ): { start: string; end: string }[] {
    const suggestions: { start: string; end: string }[] = [];
    for (const s of schedules) {
      const [sh, sm] = (s.start_time as string).split(':').map(Number);
      const [eh, em] = (s.end_time as string).split(':').map(Number);
      const baseDate = new Date(start);
      baseDate.setUTCHours(sh, sm, 0, 0);
      for (let t = sh * 60 + sm; t + 30 <= eh * 60 + em; t += 30) {
        const candStart = new Date(start);
        candStart.setUTCHours(Math.floor(t / 60), t % 60, 0, 0);
        const candEnd = new Date(candStart.getTime() + 30 * 60 * 1000);
        if (candStart > start && suggestions.length < 3) {
          suggestions.push({
            start: candStart.toISOString(),
            end: candEnd.toISOString(),
          });
        }
      }
      if (suggestions.length >= 3) break;
    }
    return suggestions;
  }
}
