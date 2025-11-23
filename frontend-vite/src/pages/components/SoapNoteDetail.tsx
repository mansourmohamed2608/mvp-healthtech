import { FunctionComponent } from 'react';

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

const SoapNoteDetail: FunctionComponent<{
  note: Note;
  onApprove: () => void;
  onReject: () => void;
  statusBadge: JSX.Element;
}> = ({ note, onApprove, onReject, statusBadge }) => {
  return (
    <div className="space-y-4">
      <div className="flex justify-between items-start">
        <div>
          <p className="text-sm text-gray-600">Session: {note.sessionId || 'N/A'}</p>
          <p className="text-sm text-gray-600">
            Patient: {note.patientId || 'N/A'} | Clinician: {note.clinicianId || 'N/A'}
          </p>
          <p className="text-sm text-gray-600">Created: {note.createdAt || 'N/A'}</p>
        </div>
        <div className="flex items-center space-x-2">
          {statusBadge}
          <button onClick={onApprove} className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700">
            Approve
          </button>
          <button onClick={onReject} className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700">
            Reject
          </button>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-800">
        <Section title="Subjective" content={note.subjective} />
        <Section title="Objective" content={note.objective} />
        <Section title="Assessment" content={note.assessment} />
        <Section title="Plan" content={note.plan} />
      </div>
    </div>
  );
};

export default SoapNoteDetail;

function Section({ title, content }: { title: string; content?: string }) {
  return (
    <div>
      <p className="font-semibold">{title}</p>
      <p className="text-gray-700 whitespace-pre-wrap">{content || 'N/A'}</p>
    </div>
  );
}
