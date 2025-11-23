import { useEffect, useState } from 'react';
import api from '../../utils/api';
import SoapNoteDetail from './SoapNoteDetail';

type Note = {
  id: string;
  status: string;
  sessionId?: string;
  patientId?: string;
  clinicianId?: string;
  createdAt?: string;
  subjective?: string;
  objective?: string;
  assessment?: string;
  plan?: string;
};

export default function SoapNotesPanel() {
  const [notes, setNotes] = useState<Note[]>([]);
  const [selected, setSelected] = useState<Note | null>(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<string>('');
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadNotes = async (status?: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getSOAPNotes(status);
      const list = res.notes || [];
      setNotes(list);
      if (selected) {
        const refreshed = list.find((n: Note) => n.id === selected.id);
        setSelected(refreshed || null);
      }
    } catch (e: any) {
      const msg = e?.message || e?.response?.data?.message || String(e);
      setError(msg);
    }
    setLoading(false);
  };

  useEffect(() => {
    loadNotes(filter);
  }, [filter]);

  const approve = async (id: string) => {
    setMessage(null); setError(null);
    try {
      await api.approveSOAPNote(id);
      setMessage('Note approved');
      await loadNotes(filter);
    } catch (e: any) {
      const msg = e?.message || e?.response?.data?.message || String(e);
      setError(msg);
    }
  };

  const reject = async (id: string) => {
    setMessage(null); setError(null);
    try {
      await api.rejectSOAPNote(id);
      setMessage('Note rejected');
      await loadNotes(filter);
    } catch (e: any) {
      const msg = e?.message || e?.response?.data?.message || String(e);
      setError(msg);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-8">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <h2 className="text-2xl font-semibold text-gray-800">SOAP Notes</h2>
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="border p-2 rounded text-sm"
          >
            <option value="">All</option>
            <option value="pending">Pending</option>
            <option value="approved">Approved</option>
            <option value="rejected">Rejected</option>
          </select>
        </div>
        <button
          onClick={() => loadNotes(filter)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
          disabled={loading}
        >
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>
      {message && <p className="text-sm text-green-700 mb-2 bg-green-50 px-3 py-1 rounded">{message}</p>}
      {error && <p className="text-sm text-red-700 mb-2 bg-red-50 px-3 py-1 rounded">{error}</p>}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="md:col-span-1 space-y-3 max-h-96 overflow-y-auto">
          {notes.map((note) => (
            <div
              key={note.id}
              className={`border rounded p-3 cursor-pointer ${selected?.id === note.id ? 'border-blue-500' : 'border-gray-200'}`}
              onClick={() => setSelected(note)}
            >
              <div className="flex justify-between items-center">
                <p className="font-semibold text-gray-900">Status</p>
                {statusBadge(note.status)}
              </div>
              <p className="text-xs text-gray-600">Session: {note.sessionId || 'N/A'}</p>
              <p className="text-xs text-gray-600">Patient: {note.patientId || 'N/A'}</p>
            </div>
          ))}
          {notes.length === 0 && <p className="text-sm text-gray-600">No notes.</p>}
        </div>
        <div className="md:col-span-2 border rounded p-4 bg-gray-50 min-h-[240px]">
          {selected ? (
            <SoapNoteDetail
              note={selected}
              onApprove={() => approve(selected.id)}
              onReject={() => reject(selected.id)}
              statusBadge={statusBadge(selected.status)}
            />
          ) : (
            <p className="text-sm text-gray-600">Select a note to view details.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function statusBadge(status: string) {
  const base = "px-2 py-1 rounded text-xs font-semibold";
  const cls =
    status === "approved"
      ? "bg-green-100 text-green-800"
      : status === "rejected"
      ? "bg-red-100 text-red-800"
      : "bg-yellow-100 text-yellow-800";
  return <span className={`${base} ${cls}`}>{status}</span>;
}
