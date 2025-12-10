import { Controller, Get, Query, UseGuards } from '@nestjs/common';
import { VaBookingService } from './va_booking.service';
import { JwtAuthGuard } from '../auth/jwt.guard';
import { Roles } from '../auth/roles.decorator';
import { Pool } from 'pg';
import { ConfigService } from '@nestjs/config';

@UseGuards(JwtAuthGuard)
@Roles('clinician')
@Controller('va')
export class VaController {
  private pool: Pool | null;
  constructor(private readonly vaService: VaBookingService, config: ConfigService) {
    const url = config.get<string>('DATABASE_URL');
    this.pool = url ? new Pool({ connectionString: url }) : null;
  }

  @Get('appointments')
  async listAppointments(@Query('doctorId') doctorId?: string, @Query('date') date?: string) {
    if (!this.pool) return { appointments: [] };
    const client = await this.pool.connect();
    try {
      const params: any[] = [];
      const conds: string[] = [];
      if (doctorId) {
        params.push(doctorId);
        conds.push(`a.doctor_id = $${params.length}`);
      }
      if (date) {
        params.push(date);
        conds.push(`a.start_datetime::date = $${params.length}`);
      }
      const where = conds.length ? `WHERE ${conds.join(' AND ')}` : '';
      const { rows } = await client.query(
        `SELECT a.id, a.patient_name, a.start_datetime, a.end_datetime, a.status, d.name as doctor_name
         FROM appointments a
         JOIN doctors d ON d.id = a.doctor_id
         ${where}
         ORDER BY a.start_datetime DESC
         LIMIT 50`,
        params,
      );
      return {
        appointments: rows.map((r) => ({
          id: r.id,
          doctor_name: r.doctor_name,
          patient_name: r.patient_name ? `${r.patient_name[0]}***` : '',
          start: r.start_datetime,
          end: r.end_datetime,
          status: r.status,
        })),
      };
    } finally {
      client.release();
    }
  }
}
